# main.py
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

# --- Configuration ---

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s: %(message)s"
)


@dataclass
class Config:
    """Centralized configuration for the bot."""

    # Secrets
    DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN")
    GEMINI_KEY: str = os.getenv("GEMINI_KEY")

    # Bot Settings
    PREFIX = ["e ", "E "]
    SNIPE_CACHE_LIMIT = 10  # Max deleted/edited messages to store per channel

    # API Settings
    GEMINI_API_URL: str = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"

    # Embed Colors
    EMBED_COLORS: Dict[str, discord.Color] = field(
        default_factory=lambda: {
            "success": discord.Color.green(),
            "error": discord.Color.red(),
            "analyse": discord.Color.orange(),
            "grammar": discord.Color.blue(),
            "summary": discord.Color.purple(),
            "solution": discord.Color.teal(),
            "snipe": discord.Color.light_grey(),
            "editsnipe": discord.Color.gold(),  # Added for editsnipe command
        }
    )


# Instantiate and validate config
config = Config()
if not config.DISCORD_TOKEN or not config.GEMINI_KEY:
    logging.critical("CRITICAL: DISCORD_TOKEN or GEMINI_KEY not found.")
    exit("Exiting: Missing required environment variables.")


# --- API Client ---


class APIError(Exception):
    """Base exception for API-related errors."""

    pass


class APIRequestError(APIError):
    """Exception for failed API requests."""

    def __init__(self, status: int, text: str):
        super().__init__(f"API request failed with status {status}: {text}")


class APIParseError(APIError):
    """Exception for errors parsing the API response."""

    def __init__(self, message: str, response: Dict[str, Any]):
        super().__init__(f"{message}\nResponse: {response}")


class GeminiAPIClient:
    """Handles all interactions with the Google Gemini API."""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = aiohttp.ClientSession(
            headers={"Content-Type": "application/json"}
        )

    async def close(self):
        """Closes the aiohttp session."""
        await self.session.close()

    async def _generate_content(
        self, prompt: str, schema: Optional[Dict] = None
    ) -> Union[list, str, None]:
        """Generic method to make a request to the Gemini API."""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        if schema:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": schema,
            }

        async with self.session.post(self.api_url, json=payload) as response:
            if not response.ok:
                raise APIRequestError(response.status, await response.text())
            response_json = await response.json()

        try:
            part = response_json["candidates"][0]["content"]["parts"][0]
            # If the response should be JSON, load it from the text field
            if schema and "text" in part:
                # The API returns a JSON string inside the 'text' field when a schema is used
                return json.loads(part["text"])
            # Otherwise, return the text directly
            return part.get("text")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise APIParseError(f"Failed to parse API response: {e}", response_json)

    async def get_fallacies(self, text: str) -> Optional[List[Dict]]:
        """Identifies logical fallacies in text."""
        prompt = f"""Analyse the following text for logical fallacies. For each fallacy found, provide its name, an explanation, and the specific quote. If none are found, return an empty array. Text: "{text}" """
        schema = {
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
        }
        return await self._generate_content(prompt, schema)

    async def get_grammar_errors(self, text: str) -> Optional[List[Dict]]:
        """Identifies grammatical errors in text."""
        prompt = f"""Analyse the text for grammatical errors. For each error, provide its type, an explanation, the suggested correction, and the quote. If none, return an empty array. Text: "{text}" """
        schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "error_type": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                    "correction": {"type": "STRING"},
                    "quote": {"type": "STRING"},
                },
                "required": ["error_type", "explanation", "correction", "quote"],
            },
        }
        return await self._generate_content(prompt, schema)

    async def get_summary(self, text: str) -> Optional[str]:
        """Generates a concise summary of a conversation."""
        prompt = f"Provide a concise summary of the following conversation, capturing the main points and arguments:\n---\n{text}\n---"
        return await self._generate_content(prompt)

    async def get_solution(self, text: str) -> Optional[str]:
        """Proposes a neutral, actionable solution for a discussion."""
        prompt = f"""Act as a neutral third-party observer. Analyse the conversation, identify the core issue (argument or problem), and propose a concise, practical, and actionable solution. Your tone should be constructive and unbiased. Conversation:\n---\n{text}\n---"""
        return await self._generate_content(prompt)


# --- Bot Cog ---


class AnalysisCog(commands.Cog):
    """Cog for text analysis commands."""

    def __init__(self, bot: commands.Bot, api_client: GeminiAPIClient):
        self.bot = bot
        self.api_client = api_client
        # In-memory caches for deleted and edited messages
        self.sniped_messages: Dict[int, List[discord.Message]] = {}
        self.edited_messages: Dict[
            int, List[Tuple[discord.Message, discord.Message]]
        ] = {}

    # --- Helper Methods ---

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncates text to a max length, adding an ellipsis if needed."""
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    @staticmethod
    def _create_embed(
        title: str, color: discord.Color, description: str = "", footer: str = ""
    ) -> discord.Embed:
        """Creates a standardized Discord embed."""
        embed = discord.Embed(title=title, description=description, color=color)
        if footer:
            embed.set_footer(text=footer)
        return embed

    async def _get_target_from_context(
        self, ctx: commands.Context, text_arg: Optional[str]
    ) -> Tuple[str, discord.Member]:
        """Gets the target text and author from context (reply or argument)."""
        if ctx.message.reference and ctx.message.reference.message_id:
            try:
                msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
                return msg.content, msg.author
            except (discord.NotFound, discord.Forbidden):
                raise commands.CommandError("Could not fetch the replied message.")
        if text_arg:
            return text_arg, ctx.author
        raise commands.UserInputError(
            "Please reply to a message or provide text directly."
        )

    async def _fetch_conversation_from_reply(
        self, ctx: commands.Context
    ) -> Tuple[str, discord.Message]:
        """Fetches conversation history starting from a replied-to message."""
        if not ctx.message.reference or not ctx.message.reference.message_id:
            raise commands.UserInputError(
                "You must reply to a message to use this command."
            )

        try:
            start_message = await ctx.channel.fetch_message(
                ctx.message.reference.message_id
            )
        except (discord.NotFound, discord.Forbidden):
            raise commands.CommandError("Could not access the starting message.")

        history = [
            msg async for msg in ctx.channel.history(after=start_message, limit=500)
        ]
        messages = [start_message] + history

        convo_text = "\n".join(
            f"{m.author.display_name}: {m.content}" for m in messages if m.content
        )
        if not convo_text:
            raise commands.CommandError(
                "There is no text in this conversation to analyze."
            )

        return convo_text, start_message

    # --- Event Listeners ---

    @commands.Cog.listener()
    async def on_ready(self):
        """Called when the bot successfully connects."""
        logging.info(f"Logged in as {self.bot.user}. Bot is ready.")
        print("-------------------------------------------------")

    @commands.Cog.listener()
    async def on_message_delete(self, message: discord.Message):
        """Caches a deleted message for the snipe command."""
        # Ignore messages from bots and those without content/attachments
        if message.author.bot or (not message.content and not message.attachments):
            return

        channel_id = message.channel.id
        # Initialize the list for the channel if it doesn't exist
        if channel_id not in self.sniped_messages:
            self.sniped_messages[channel_id] = []

        # Add the message to the front of the list
        self.sniped_messages[channel_id].insert(0, message)

        # Trim the list to the configured limit
        self.sniped_messages[channel_id] = self.sniped_messages[channel_id][
            : config.SNIPE_CACHE_LIMIT
        ]

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Caches an edited message for the editsnipe command."""
        # Ignore edits from bots, or if the content is the same (e.g., embed added)
        if before.author.bot or before.content == after.content:
            return

        channel_id = before.channel.id
        # Initialize the list for the channel if it doesn't exist
        if channel_id not in self.edited_messages:
            self.edited_messages[channel_id] = []

        # Add the message tuple (before, after) to the front of the list
        self.edited_messages[channel_id].insert(0, (before, after))

        # Trim the list to the configured limit
        self.edited_messages[channel_id] = self.edited_messages[channel_id][
            : config.SNIPE_CACHE_LIMIT
        ]

    @commands.Cog.listener()
    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ):
        """Generic error handler for the cog."""
        # Silently ignore commands that are not found
        if isinstance(error, commands.CommandNotFound):
            return

        # Handle known user input errors
        if isinstance(error, (commands.UserInputError, commands.CommandError)):
            embed = self._create_embed(
                "Error", config.EMBED_COLORS["error"], description=f"⚠️ {error}"
            )
            await ctx.send(embed=embed)
            return

        # For all other unexpected errors, log the full traceback but send a generic message
        logging.exception(
            f"Unexpected error in command '{ctx.command}':", exc_info=error
        )
        embed = self._create_embed(
            "An Unexpected Error Occurred",
            config.EMBED_COLORS["error"],
            description="Sorry, something went wrong. The details have been logged for the developer.",
        )
        await ctx.send(embed=embed)

    # --- Commands ---

    @commands.command(
        name="analyse", aliases=["analyze"], help="Analyses text for logical fallacies."
    )
    async def analyse(self, ctx: commands.Context, *, text: Optional[str] = None):
        """Analyses text from a reply or argument for logical fallacies."""
        target_text, author = await self._get_target_from_context(ctx, text)

        async with ctx.typing():
            fallacies = await self.api_client.get_fallacies(target_text)

        if not fallacies:
            embed = self._create_embed(
                "Analysis Complete",
                config.EMBED_COLORS["success"],
                "✅ No logical fallacies were detected.",
            )
            await ctx.send(embed=embed)
            return

        embed = self._create_embed(
            "Logical Fallacy Analysis",
            config.EMBED_COLORS["analyse"],
            f"Found {len(fallacies)} potential fallac{'y' if len(fallacies) == 1 else 'ies'}:",
            footer=f"Analysed for {author.display_name}",
        )
        for i, fallacy in enumerate(fallacies, 1):
            val = f'**Explanation:** {fallacy.get("explanation", "N/A")}\n**Quote:** *"{fallacy.get("quote", "N/A")}"*'
            embed.add_field(
                name=f"{i}. {fallacy.get('fallacy_name', 'Unknown')}",
                value=self._truncate(val, 1024),
                inline=False,
            )

        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(
        name="grammar", help="Checks text for grammar and spelling errors."
    )
    async def grammar(self, ctx: commands.Context, *, text: Optional[str] = None):
        """Checks text from a reply or argument for grammatical errors."""
        target_text, author = await self._get_target_from_context(ctx, text)

        async with ctx.typing():
            errors = await self.api_client.get_grammar_errors(target_text)

        if not errors:
            embed = self._create_embed(
                "Grammar Check Complete",
                config.EMBED_COLORS["success"],
                "✅ No grammatical errors were detected.",
            )
            await ctx.send(embed=embed)
            return

        embed = self._create_embed(
            "Grammar Analysis",
            config.EMBED_COLORS["grammar"],
            f"Found {len(errors)} potential error{'s' if len(errors) > 1 else ''}:",
            footer=f"Checked for {author.display_name}",
        )
        for i, error in enumerate(errors, 1):
            val = f'**Explanation:** {error.get("explanation", "N/A")}\n**Correction:** `{error.get("correction", "N/A")}`\n**Original:** *"{error.get("quote", "N/A")}"*'
            embed.add_field(
                name=f"{i}. {error.get('error_type', 'Unknown')}",
                value=self._truncate(val, 1024),
                inline=False,
            )

        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(
        name="tldr",
        aliases=["summarise", "summarize"],
        help="Summarises a conversation.",
    )
    async def tldr(self, ctx: commands.Context):
        """Summarises a conversation starting from the replied-to message."""
        convo_text, start_message = await self._fetch_conversation_from_reply(ctx)

        async with ctx.typing():
            summary = await self.api_client.get_summary(convo_text)

        if not summary:
            raise commands.CommandError("Failed to generate a summary.")

        embed = self._create_embed(
            "Conversation Summary (TL;DR)",
            config.EMBED_COLORS["summary"],
            description=self._truncate(summary, 4096),
            footer=f"Summary of conversation since {start_message.author.display_name}'s message.",
        )
        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(
        name="solution", help="Gives a neutral solution on a conversation."
    )
    async def solution(self, ctx: commands.Context):
        """Provides a neutral solution for a conversation starting from the replied-to message."""
        convo_text, start_message = await self._fetch_conversation_from_reply(ctx)

        async with ctx.typing():
            solution_text = await self.api_client.get_solution(convo_text)

        if not solution_text:
            raise commands.CommandError("Failed to generate a solution.")

        embed = self._create_embed(
            "A Potential Solution",
            config.EMBED_COLORS["solution"],
            description=self._truncate(solution_text, 4096),
            footer=f"Solution for conversation since {start_message.author.display_name}'s message.",
        )
        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(name="snipe", help="Displays the last message that was deleted.")
    async def snipe(self, ctx: commands.Context, index: int = 1):
        """Shows the last deleted message in the channel."""
        channel_id = ctx.channel.id

        # Check if there are any messages sniped in this channel
        if (
            channel_id not in self.sniped_messages
            or not self.sniped_messages[channel_id]
        ):
            raise commands.CommandError(
                "There are no messages to snipe in this channel."
            )

        # Validate the index
        if index <= 0:
            raise commands.UserInputError(
                "Index must be a positive number (1, 2, etc.)."
            )

        sniped_list = self.sniped_messages[channel_id]
        if index > len(sniped_list):
            raise commands.CommandError(
                f"There are only {len(sniped_list)} deleted messages stored. Cannot snipe message #{index}."
            )

        # Get the message from our cache (index-1 because user input is 1-based)
        msg = sniped_list[index - 1]

        # Create the embed
        embed = self._create_embed(
            "Sniped Message",
            config.EMBED_COLORS["snipe"],
            description=msg.content or "_(No text content)_",
            footer=f"Sniping message {index}/{len(sniped_list)}.",
        )
        embed.set_author(name=str(msg.author), icon_url=msg.author.display_avatar.url)
        embed.timestamp = msg.created_at  # Show when the original message was sent

        # If there was an image/attachment, display it
        if msg.attachments and msg.attachments[0].url:
            embed.set_image(url=msg.attachments[0].url)

        await ctx.send(embed=embed)

    @commands.command(
        name="editsnipe", help="Displays the last message that was edited."
    )
    async def editsnipe(self, ctx: commands.Context, index: int = 1):
        """Shows the last edited message in the channel."""
        channel_id = ctx.channel.id

        # Check if there are any messages to snipe
        if (
            channel_id not in self.edited_messages
            or not self.edited_messages[channel_id]
        ):
            raise commands.CommandError(
                "There are no recently edited messages to snipe in this channel."
            )

        # Validate the index
        if index <= 0:
            raise commands.UserInputError(
                "Index must be a positive number (1, 2, etc.)."
            )

        edited_list = self.edited_messages[channel_id]
        if index > len(edited_list):
            raise commands.CommandError(
                f"There are only {len(edited_list)} edited messages stored. Cannot snipe edit #{index}."
            )

        # Get the message tuple from our cache
        before_msg, after_msg = edited_list[index - 1]

        # Create the embed
        embed = self._create_embed(
            "Edited Message",
            config.EMBED_COLORS["editsnipe"],
            description=f"[Jump to Message]({after_msg.jump_url})",
            footer=f"Sniping edit {index}/{len(edited_list)}.",
        )
        embed.set_author(
            name=str(after_msg.author), icon_url=after_msg.author.display_avatar.url
        )
        embed.timestamp = after_msg.edited_at  # Show when the message was edited

        # Add fields for before and after content
        embed.add_field(
            name="Before",
            value=self._truncate(before_msg.content or "_(No text)_", 1024),
            inline=False,
        )
        embed.add_field(
            name="After",
            value=self._truncate(after_msg.content or "_(No text)_", 1024),
            inline=False,
        )

        await ctx.send(embed=embed)


# --- Main Execution ---


async def main():
    """Sets up the bot and runs it."""
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True
    intents.guilds = True  # Needed for on_message events to work reliably

    bot = commands.Bot(command_prefix=config.PREFIX, intents=intents)

    # Pass the full URL from the config to the API client
    api_client = GeminiAPIClient(config.GEMINI_API_URL)

    # Add cogs and other setup here
    await bot.add_cog(AnalysisCog(bot, api_client))

    try:
        await bot.start(config.DISCORD_TOKEN)
    finally:
        await api_client.close()


if __name__ == "__main__":
    asyncio.run(main())
