import discord
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# It's highly recommended to use environment variables for your tokens
# Create a file named .env and add the following line:
# DISCORD_TOKEN=YOUR_BOT_TOKEN_HERE
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

# The Gemini API Key is not needed here as it's provided by the environment
# where the request is made. If running locally outside that environment,
# you would need to get a key from Google AI Studio.
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"

# Define the bot's intents
# We need message_content to read the user's arguments
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    """
    This function is called when the bot successfully connects to Discord.
    """
    print(f"We have logged in as {client.user}")
    print("Bot is ready to analyse arguments and grammar!")
    print("Reply to a message with e analyse or use e analyse <your argument>.")
    print("Reply to a message with e grammar or use e grammar <your text>.")
    print(
        "Reply to a message with e tldr to summarise the conversation since that message."
    )
    print("-------------------------------------------------")


@client.event
async def on_message(message):
    """
    This function is called every time a message is sent in a channel the bot can see.
    """
    # Ignore messages sent by the bot itself to prevent loops
    if message.author == client.user:
        return

    # Check if the message starts with the e analyse command
    if message.content.lower().startswith("e analyse"):
        argument_text = ""
        author_of_argument = None

        # Check if the command is a reply to another message
        if message.reference and message.reference.message_id:
            try:
                # Fetch the message that this one is replying to
                replied_to_message = await message.channel.fetch_message(
                    message.reference.message_id
                )
                argument_text = replied_to_message.content.strip()
                author_of_argument = replied_to_message.author
            except discord.NotFound:
                await message.channel.send(
                    "I couldn't find the message you replied to."
                )
                return
            except discord.Forbidden:
                await message.channel.send(
                    "I don't have permission to read the message history here."
                )
                return
        else:
            # It's not a reply, so extract text from the command itself
            argument_text = message.content[len("e analyse") :].strip()
            author_of_argument = message.author

        if not argument_text:
            await message.channel.send(
                "Please provide an argument. Reply to a message with `e analyse` or use `e analyse <your argument>`."
            )
            return

        # Let the user know the analysis is in progress
        processing_message = await message.channel.send(
            f"🤔 Analyzing the argument from {author_of_argument.mention}..."
        )

        try:
            # Call the function to get fallacies from the Gemini API
            fallacies = await get_fallacies(argument_text)

            # Delete the "Analyzing..." message
            await processing_message.delete()

            if not fallacies:
                # If no fallacies are found, send a confirmation message
                embed = discord.Embed(
                    title="Analysis Complete",
                    description="✅ No logical fallacies were detected in the provided text.",
                    color=discord.Color.green(),
                )
                await message.channel.send(embed=embed)
            else:
                # If fallacies are found, create an embed to display them
                embed = discord.Embed(
                    title="Logical Fallacy Analysis",
                    description=f"Found {len(fallacies)} potential fallac{'y' if len(fallacies) == 1 else 'ies'} in the argument:",
                    color=discord.Color.orange(),
                )
                embed.set_footer(text=f"Analysed for {author_of_argument.display_name}")

                for i, fallacy in enumerate(fallacies, 1):
                    # Add a field for each fallacy found
                    field_name = (
                        f"{i}. {fallacy.get('fallacy_name', 'Unknown Fallacy')}"
                    )
                    field_value = (
                        f"**Explanation:** {fallacy.get('explanation', 'No explanation provided.')}\n"
                        f'**Quote:** *"{fallacy.get("quote", "N/A")}"*'
                    )

                    # MODIFIED: Truncate the field value if it exceeds Discord's limit of 1024 characters.
                    if len(field_value) > 1024:
                        field_value = field_value[:1021] + "..."

                    embed.add_field(name=field_name, value=field_value, inline=False)

                # Reply to the original command message with the results
                await message.reply(embed=embed, mention_author=False)

        except Exception as e:
            print(f"An error occurred: {e}")
            # Delete the "Analyzing..." message and send an error message
            await processing_message.delete()
            error_embed = discord.Embed(
                title="Analysis Failed",
                description="Sorry, something went wrong while trying to analyse the argument. Please try again later.",
                color=discord.Color.red(),
            )
            await message.channel.send(embed=error_embed)

    # Check if the message starts with the e grammar command
    elif message.content.lower().startswith("e grammar"):
        text_to_check = ""
        author_of_text = None

        # Check if the command is a reply to another message
        if message.reference and message.reference.message_id:
            try:
                # Fetch the message that this one is replying to
                replied_to_message = await message.channel.fetch_message(
                    message.reference.message_id
                )
                text_to_check = replied_to_message.content.strip()
                author_of_text = replied_to_message.author
            except discord.NotFound:
                await message.channel.send(
                    "I couldn't find the message you replied to."
                )
                return
            except discord.Forbidden:
                await message.channel.send(
                    "I don't have permission to read the message history here."
                )
                return
        else:
            # It's not a reply, so extract text from the command itself
            text_to_check = message.content[len("e grammar") :].strip()
            author_of_text = message.author

        if not text_to_check:
            await message.channel.send(
                "Please provide text to check. Reply to a message with `e grammar` or use `e grammar <your text>`."
            )
            return

        # Let the user know the check is in progress
        processing_message = await message.channel.send(
            f"✍️ Checking grammar for {author_of_text.mention}..."
        )

        try:
            # Call the new function to get grammar errors from the Gemini API
            errors = await get_grammar_errors(text_to_check)

            # Delete the "Checking..." message
            await processing_message.delete()

            if not errors:
                # If no errors are found, send a confirmation message
                embed = discord.Embed(
                    title="Grammar Check Complete",
                    description="✅ No grammatical errors were detected.",
                    color=discord.Color.green(),
                )
                await message.channel.send(embed=embed)
            else:
                # If errors are found, create an embed to display them
                embed = discord.Embed(
                    title="Grammar Analysis",
                    description=f"Found {len(errors)} potential error{'s' if len(errors) > 1 else ''}:",
                    color=discord.Color.blue(),
                )
                embed.set_footer(text=f"Checked for {author_of_text.display_name}")

                for i, error in enumerate(errors, 1):
                    # Add a field for each error found
                    field_name = f"{i}. {error.get('error_type', 'Unknown Error')}"
                    field_value = (
                        f"**Explanation:** {error.get('explanation', 'No explanation provided.')}\n"
                        f"**Correction:** `{error.get('correction', 'N/A')}`\n"
                        f'**Original:** *"{error.get("quote", "N/A")}"*'
                    )

                    # MODIFIED: Truncate the field value if it exceeds Discord's limit of 1024 characters.
                    if len(field_value) > 1024:
                        field_value = field_value[:1021] + "..."

                    embed.add_field(name=field_name, value=field_value, inline=False)

                # Reply to the original command message with the results
                await message.reply(embed=embed, mention_author=False)

        except Exception as e:
            print(f"An error occurred: {e}")
            # Delete the "Checking..." message and send an error message
            await processing_message.delete()
            error_embed = discord.Embed(
                title="Grammar Check Failed",
                description="Sorry, something went wrong while trying to check the grammar. Please try again later.",
                color=discord.Color.red(),
            )
            await message.channel.send(embed=error_embed)

    # Handle 'e tldr' command
    elif message.content.lower() == "e tldr":
        # This command must be a reply to a message
        if not message.reference or not message.reference.message_id:
            await message.channel.send(
                "You must reply to the starting message of the conversation you want to summarize with `e tldr`."
            )
            return

        processing_message = await message.channel.send(
            "📚 Summarizing the conversation..."
        )

        try:
            # Fetch the starting message
            start_message = await message.channel.fetch_message(
                message.reference.message_id
            )

            # Fetch all messages between the start message and the command message
            messages_to_summarize = [start_message]
            # We use history to get messages after the starting one, up until the current one.
            async for msg in message.channel.history(
                after=start_message, before=message
            ):
                messages_to_summarize.append(msg)

            # Format the conversation for the API
            conversation_text = "\n".join(
                f"{msg.author.display_name}: {msg.content}"
                for msg in messages_to_summarize
            )

            if not conversation_text:
                await processing_message.edit(content="There's nothing to summarize!")
                return

            # Get the summary from the API
            summary = await get_summary(conversation_text)

            await processing_message.delete()

            # MODIFIED: Truncate the summary if it exceeds Discord's description limit of 4096 characters.
            if len(summary) > 4096:
                summary = summary[:4093] + "..."

            # Create and send the embed with the summary
            embed = discord.Embed(
                title="Conversation Summary (TL;DR)",
                description=summary,
                color=discord.Color.purple(),
            )
            embed.set_footer(
                text=f"Summary of conversation since {start_message.author.display_name}'s message."
            )
            await message.reply(embed=embed, mention_author=False)

        except discord.NotFound:
            await processing_message.delete()
            await message.channel.send(
                "I couldn't find the starting message you replied to."
            )
        except discord.Forbidden:
            await processing_message.delete()
            await message.channel.send(
                "I don't have permission to read the message history here."
            )
        except Exception as e:
            print(f"An error occurred during summarization: {e}")
            await processing_message.delete()
            error_embed = discord.Embed(
                title="Summarization Failed",
                description="Sorry, something went wrong. Please try again later.",
                color=discord.Color.red(),
            )
            await message.channel.send(embed=error_embed)


async def get_fallacies(text_to_analyse: str) -> list:
    """
    Sends the text to the Gemini API to detect logical fallacies.
    """
    prompt = f"""
        analyse the following text for logical fallacies. For each fallacy you find, provide:
        1. The name of the fallacy.
        2. A brief explanation of why it is that fallacy in the context of the text.
        3. The specific quote from the text that contains the fallacy.

        Text to analyse: "{text_to_analyse}"
    """

    # The payload structure for the Gemini API call
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
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

    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        # Safe navigation through the JSON response
        try:
            part = result["candidates"][0]["content"]["parts"][0]
            if "text" in part:
                return json.loads(part["text"])
            else:
                return []
        except (KeyError, IndexError):
            return []  # Return empty list if the expected structure is not present
    else:
        raise Exception(
            f"API request failed with status {response.status_code}: {response.text}"
        )


async def get_grammar_errors(text_to_analyse: str) -> list:
    """
    Sends the text to the Gemini API to detect grammatical errors.
    """
    prompt = f"""
        Analyse the following text for grammatical errors, including spelling and punctuation mistakes. For each error you find, provide:
        1. The type of error (e.g., "Spelling", "Punctuation", "Subject-Verb Agreement").
        2. A brief explanation of the mistake.
        3. The suggested correction.
        4. The specific quote from the text that contains the error.

        If there are no errors, return an empty array.

        Text to analyse: "{text_to_analyse}"
    """

    # The payload structure for the Gemini API call
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
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
                    "required": ["error_type", "explanation", "correction", "quote"],
                },
            },
        },
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        # Safe navigation through the JSON response
        try:
            part = result["candidates"][0]["content"]["parts"][0]
            if "text" in part:
                return json.loads(part["text"])
            else:
                return []
        except (KeyError, IndexError):
            return []  # Return empty list if the expected structure is not present
    else:
        raise Exception(
            f"API request failed with status {response.status_code}: {response.text}"
        )


async def get_summary(conversation_text: str) -> str:
    """
    Sends a conversation to the Gemini API and asks for a summary.
    """
    prompt = f"""
    Please provide a concise summary of the following conversation. 
    Capture the main points, key arguments, and the overall conclusion or outcome if one exists.
    
    Conversation to summarize:
    ---
    {conversation_text}
    ---
    """

    # The payload for a simple text response
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Could not parse the summary from the API response."
    else:
        raise Exception(
            f"API request failed with status {response.status_code}: {response.text}"
        )


# Check if the token is set before running the bot
if DISCORD_TOKEN:
    client.run(DISCORD_TOKEN)
else:
    print("Error: DISCORD_TOKEN not found.")
    print("Please create a .env file and add your bot token.")
    print("Example: DISCORD_TOKEN=YOUR_BOT_TOKEN_HERE")
