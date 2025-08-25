# main.py
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
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_KEY = os.getenv('GEMINI_KEY')

# The Gemini API Key is not needed here as it's provided by the environment
# where the request is made. If running locally outside that environment,
# you would need to get a key from Google AI Studio.
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_KEY}"

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
    print(f'We have logged in as {client.user}')
    print('Bot is ready to analyse arguments!')
    print('Reply to a message with e analyse or use e analyse <your argument>.')
    print('-------------------------------------------------')

@client.event
async def on_message(message):
    """
    This function is called every time a message is sent in a channel the bot can see.
    """
    # Ignore messages sent by the bot itself to prevent loops
    if message.author == client.user:
        return

    # Check if the message starts with the e analyse command
    if message.content.startswith('e analyse'):
        argument_text = ""
        author_of_argument = None

        # NEW: Check if the command is a reply to another message
        if message.reference and message.reference.message_id:
            try:
                # Fetch the message that this one is replying to
                replied_to_message = await message.channel.fetch_message(message.reference.message_id)
                argument_text = replied_to_message.content.strip()
                author_of_argument = replied_to_message.author
            except discord.NotFound:
                await message.channel.send("I couldn't find the message you replied to.")
                return
            except discord.Forbidden:
                await message.channel.send("I don't have permission to read the message history here.")
                return
        else:
            # OLD BEHAVIOR: It's not a reply, so extract text from the command itself
            argument_text = message.content[len('e analyse'):].strip()
            author_of_argument = message.author

        if not argument_text:
            await message.channel.send("Please provide an argument. Reply to a message with `e analyse` or use `e analyse <your argument>`.")
            return

        # Let the user know the analysis is in progress
        processing_message = await message.channel.send(f"🤔 Analyzing the argument from {author_of_argument.mention}...")

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
                    color=discord.Color.green()
                )
                await message.channel.send(embed=embed)
            else:
                # If fallacies are found, create an embed to display them
                embed = discord.Embed(
                    title="Logical Fallacy Analysis",
                    description=f"Found {len(fallacies)} potential fallac{'y' if len(fallacies) == 1 else 'ies'} in the argument:",
                    color=discord.Color.orange()
                )
                embed.set_footer(text=f"analysed for {author_of_argument.display_name}")

                for i, fallacy in enumerate(fallacies, 1):
                    # Add a field for each fallacy found
                    field_name = f"{i}. {fallacy.get('fallacy_name', 'Unknown Fallacy')}"
                    field_value = (
                        f"**Explanation:** {fallacy.get('explanation', 'No explanation provided.')}\n"
                        f"**Quote:** *\"{fallacy.get('quote', 'N/A')}\"*"
                    )
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
                color=discord.Color.red()
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
                        "fallacy_name": { "type": "STRING" },
                        "explanation": { "type": "STRING" },
                        "quote": { "type": "STRING" }
                    },
                    "required": ["fallacy_name", "explanation", "quote"]
                }
            }
        }
    }

    headers = {'Content-Type': 'application/json'}

    # Using requests library for the API call.
    # In a larger, more complex bot, you might use an async library like aiohttp.
    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        part = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
        if 'text' in part:
            return json.loads(part['text'])
        else:
            # This can happen if the model finds no fallacies and returns an empty response
            return []
    else:
        # Raise an exception if the API call was not successful
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

# Check if the token is set before running the bot
if DISCORD_TOKEN:
    client.run(DISCORD_TOKEN)
else:
    print("Error: DISCORD_TOKEN not found.")
    print("Please create a .env file and add your bot token.")
    print("Example: DISCORD_TOKEN=YOUR_BOT_TOKEN_HERE")
