import discord
from discord.ext import commands
import os
import aiohttp
import json
import logging
from dotenv import load_dotenv
from typing import Optional, Union

# --- Configuration and Setup ---

# Load environment variables from a .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s: %(message)s"
)

# It's highly recommended to use environment variables for your tokens
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not DISCORD_TOKEN or not GEMINI_KEY:
    logging.critical(
        "CRITICAL: DISCORD_TOKEN or GEMINI_KEY not found in environment variables."
    )
    exit("Exiting: Missing required environment variables.")

# Define constants
PREFIX = "e "
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
EMBED_COLOR_SUCCESS = discord.Color.green()
EMBED_COLOR_ERROR = discord.Color.red()
EMBED_COLOR_ANALYSE = discord.Color.orange()
EMBED_COLOR_GRAMMAR = discord.Color.blue()
EMBED_COLOR_SUMMARY = discord.Color.purple()


# --- Bot Definition ---

# Define the bot's intents. We need message_content to read the user's arguments.
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)


# --- API Interaction Logic ---


class GeminiAPI:
    """Handles all interactions with the Google Gemini API."""

    @staticmethod
    async def _make_request(payload: dict) -> dict:
        """Makes an asynchronous POST request to the Gemini API."""
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GEMINI_API_URL, headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    logging.error(
                        f"API request failed with status {response.status}: {text}"
                    )
                    raise commands.CommandError(
                        "The API request failed. Please check the logs."
                    )

    @staticmethod
    def _parse_response(result: dict, data_key: str = "text") -> Union[list, str, None]:
        """Safely parses the JSON response from the Gemini API."""
        try:
            part = result["candidates"][0]["content"]["parts"][0]
            if data_key in part:
                # If expecting a JSON string, load it
                if isinstance(part[data_key], str) and part[
                    data_key
                ].strip().startswith(("[", "{")):
                    return json.loads(part[data_key])
                return part[data_key]
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logging.error(f"Failed to parse API response: {e}\nResponse: {result}")
            return None

    async def get_fallacies(self, text: str) -> Optional[list]:
        """Gets logical fallacies from a given text."""
        prompt = f"""
        Analyse the following text for logical fallacies. For each fallacy you find, provide:
        1. The name of the fallacy.
        2. A brief explanation of why it is that fallacy in the context of the text.
        3. The specific quote from the text that contains the fallacy.
        If no fallacies are found, return an empty array. 
        Text to analyse: "{text}"
        """
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "fallacy_name": {"type": "STRING"},
                            "explanation": {"type": "STRING"},
                            "quote": {"type": "STRING"},
                        },
                        "required": ["fallacy_name", "explanation", "quote"],
                    },
                },
            },
        }
        response_json = await self._make_request(payload)
        return self._parse_response(response_json)

    async def get_grammar_errors(self, text: str) -> Optional[list]:
        """Gets grammatical errors from a given text."""
        prompt = f"""
        Analyse the following text for grammatical errors. For each error you find, provide:
        1. The type of error (e.g., "Spelling", "Punctuation").
        2. A brief explanation of the mistake.
        3. The suggested correction.
        4. The specific quote from the text that contains the error.
        If there are no errors, return an empty array.
        Text to analyse: "{text}"
        """
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "error_type": {"type": "STRING"},
                            "explanation": {"type": "STRING"},
                            "correction": {"type": "STRING"},
                            "quote": {"type": "STRING"},
                        },
                        "required": [
                            "error_type",
                            "explanation",
                            "correction",
                            "quote",
                        ],
                    },
                },
            },
        }
        response_json = await self._make_request(payload)
        return self._parse_response(response_json)

    async def get_summary(self, text: str) -> Optional[str]:
        """Gets a summary for a given text."""
        prompt = f"""
        Provide a concise summary of the following conversation.
        Capture the main points, key arguments, and the overall outcome if one exists.
        Conversation:
        ---
        {text}
        ---
        """
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response_json = await self._make_request(payload)
        return self._parse_response(response_json)


# --- Bot Cogs (Commands and Events) ---


class AnalysisCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.api = GeminiAPI()

    @staticmethod
    def truncate_text(text: str, max_length: int) -> str:
        """Truncates text to a specified max length, adding an ellipsis if needed."""
        return text if len(text) <= max_length else text[: max_length - 3] + "..."

    async def _get_target_text_and_author(
        self, ctx: commands.Context, text_argument: Optional[str]
    ) -> tuple:
        """Helper to get text and author from a reply or command argument."""
        if ctx.message.reference and ctx.message.reference.message_id:
            try:
                replied_message = await ctx.channel.fetch_message(
                    ctx.message.reference.message_id
                )
                return replied_message.content, replied_message.author
            except (discord.NotFound, discord.Forbidden) as e:
                raise commands.CommandError(f"Could not fetch the replied message: {e}")
        elif text_argument:
            return text_argument, ctx.author
        else:
            return None, None

    @commands.Cog.listener()
    async def on_ready(self):
        """Called when the bot successfully connects to Discord."""
        logging.info(f"Logged in as {self.bot.user}")
        logging.info("Bot is ready for analysis!")
        print("-------------------------------------------------")

    @commands.command(
        name="analyse", aliases=["analyze"], help="Analyses text for logical fallacies."
    )
    async def analyse(self, ctx: commands.Context, *, text: Optional[str] = None):
        """
        Analyses text for logical fallacies.
        Usage: Reply to a message with 'e analyse' or use 'e analyse <your argument>'.
        """
        argument_text, author = await self._get_target_text_and_author(ctx, text)
        if not argument_text:
            raise commands.UserInputError("Please provide an argument to analyse.")

        async with ctx.typing():
            fallacies = await self.api.get_fallacies(argument_text)

        if fallacies is None:
            raise commands.CommandError("Failed to get a valid response from the API.")

        if not fallacies:
            embed = discord.Embed(
                title="Analysis Complete",
                description="✅ No logical fallacies were detected.",
                color=EMBED_COLOR_SUCCESS,
            )
            await ctx.send(embed=embed)
            return

        embed = discord.Embed(
            title="Logical Fallacy Analysis",
            description=f"Found {len(fallacies)} potential fallac{'y' if len(fallacies) == 1 else 'ies'}:",
            color=EMBED_COLOR_ANALYSE,
        )
        embed.set_footer(text=f"Analysed for {author.display_name}")

        for i, fallacy in enumerate(fallacies, 1):
            field_value = (
                f"**Explanation:** {fallacy.get('explanation', 'N/A')}\n"
                f'**Quote:** *"{fallacy.get("quote", "N/A")}"*'
            )
            embed.add_field(
                name=f"{i}. {fallacy.get('fallacy_name', 'Unknown')}",
                value=self.truncate_text(field_value, 1024),
                inline=False,
            )

        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(
        name="grammar", help="Checks text for grammar and spelling errors."
    )
    async def grammar(self, ctx: commands.Context, *, text: Optional[str] = None):
        """
        Checks text for grammar and spelling errors.
        Usage: Reply to a message with 'e grammar' or use 'e grammar <your text>'.
        """
        text_to_check, author = await self._get_target_text_and_author(ctx, text)
        if not text_to_check:
            raise commands.UserInputError("Please provide text to check.")

        async with ctx.typing():
            errors = await self.api.get_grammar_errors(text_to_check)

        if errors is None:
            raise commands.CommandError("Failed to get a valid response from the API.")

        if not errors:
            embed = discord.Embed(
                title="Grammar Check Complete",
                description="✅ No grammatical errors were detected.",
                color=EMBED_COLOR_SUCCESS,
            )
            await ctx.send(embed=embed)
            return

        embed = discord.Embed(
            title="Grammar Analysis",
            description=f"Found {len(errors)} potential error{'s' if len(errors) > 1 else ''}:",
            color=EMBED_COLOR_GRAMMAR,
        )
        embed.set_footer(text=f"Checked for {author.display_name}")

        for i, error in enumerate(errors, 1):
            field_value = (
                f"**Explanation:** {error.get('explanation', 'N/A')}\n"
                f"**Correction:** `{error.get('correction', 'N/A')}`\n"
                f'**Original:** *"{error.get("quote", "N/A")}"*'
            )
            embed.add_field(
                name=f"{i}. {error.get('error_type', 'Unknown Error')}",
                value=self.truncate_text(field_value, 1024),
                inline=False,
            )

        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(
        name="tldr", help="Summarises a conversation from a replied-to message."
    )
    async def tldr(self, ctx: commands.Context):
        """
        Summarises a conversation.
        Usage: Reply to the starting message of a conversation with 'e tldr'.
        """
        if not ctx.message.reference or not ctx.message.reference.message_id:
            raise commands.UserInputError(
                "You must reply to the starting message to summarize a conversation."
            )

        try:
            start_message = await ctx.channel.fetch_message(
                ctx.message.reference.message_id
            )
        except (discord.NotFound, discord.Forbidden):
            raise commands.CommandError(
                "Could not find or access the starting message."
            )

        async with ctx.typing():
            history = [
                msg async for msg in ctx.channel.history(after=start_message, limit=200)
            ]
            messages_to_summarize = [start_message] + history

            conversation_text = "\n".join(
                f"{msg.author.display_name}: {msg.content}"
                for msg in messages_to_summarize
                if msg.content
            )

            if not conversation_text:
                await ctx.send("There's nothing to summarize!")
                return

            summary = await self.api.get_summary(conversation_text)

        if not summary:
            raise commands.CommandError("Failed to generate a summary.")

        embed = discord.Embed(
            title="Conversation Summary (TL;DR)",
            description=self.truncate_text(summary, 4096),
            color=EMBED_COLOR_SUMMARY,
        )
        embed.set_footer(
            text=f"Summary of conversation since {start_message.author.display_name}'s message."
        )
        await ctx.reply(embed=embed, mention_author=False)

    @commands.Cog.listener()
    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ):
        """Generic error handler for the cog."""
        if isinstance(error, (commands.UserInputError, commands.CommandError)):
            # Send user-facing errors directly to the channel
            embed = discord.Embed(
                title="Error", description=f"⚠️ {error}", color=EMBED_COLOR_ERROR
            )
            await ctx.send(embed=embed)
        elif isinstance(error, commands.CommandNotFound):
            # Optionally, uncomment the next line to ignore 'command not found' errors silently
            # return
            pass
        else:
            # For unexpected errors, log them and notify the user
            logging.exception(
                f"An unexpected error occurred in command '{ctx.command}':",
                exc_info=error,
            )
            embed = discord.Embed(
                title="An Unexpected Error Occurred",
                description="Sorry, something went wrong. This has been logged for the developer.",
                color=EMBED_COLOR_ERROR,
            )
            await ctx.send(embed=embed)


# --- Main Execution ---


async def main():
    """Main function to run the bot."""
    async with bot:
        await bot.add_cog(AnalysisCog(bot))
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
