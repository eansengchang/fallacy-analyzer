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


# --- Main Execution ---


async def main():
    """Sets up the bot and runs it."""
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True

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