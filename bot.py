#!/usr/bin/env python3
import os
import time
import logging
import requests
import json
from dotenv import load_dotenv
from atproto import Client, models
from atproto.exceptions import AtProtocolError
from atproto_client.models.app.bsky.notification.list_notifications import (
    Params as ListNotificationsParams,
)
from atproto_client.models.app.bsky.feed.get_post_thread import (
    Params as GetPostThreadParams,
)
from atproto_client.models.app.bsky.feed.search_posts import (
    Params as SearchPostsParams,
)

# ----------- Configuration -----------
# Set this to True to reply to ALL posts mentioning @gork across Bluesky
# Set to False to only reply to direct notifications (mentions/replies to your bot)
REPLY_TO_ALL_MENTIONS = True  # Comment out this line or set to False to disable

# What to search for in posts (the mention text)
SEARCH_TERM = "@gork"  # This is what we'll search for across Bluesky

# ----------- Persistent cache files -----------
PROCESSED_URIS_FILE = "processed_uris.txt"
THREAD_REPLIES_FILE = "thread_replies.json"
MAX_REPLIES_PER_THREAD = 10

def load_processed_uris():
    """Load processed notification URIs from a local file."""
    if not os.path.exists(PROCESSED_URIS_FILE):
        return set()
    with open(PROCESSED_URIS_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def append_processed_uri(uri):
    """Append a newly processed URI to the persistent file."""
    with open(PROCESSED_URIS_FILE, "a") as f:
        f.write(f"{uri}\n")

def load_thread_replies():
    """Load thread reply counts from JSON file."""
    if not os.path.exists(THREAD_REPLIES_FILE):
        return {}
    try:
        with open(THREAD_REPLIES_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_thread_replies(thread_counts):
    """Save thread reply counts to JSON file."""
    with open(THREAD_REPLIES_FILE, "w") as f:
        json.dump(thread_counts, f, indent=2)

def get_thread_root_uri(client, uri):
    """Get the root URI of a thread to use as a unique identifier."""
    try:
        params = GetPostThreadParams(uri=uri)
        thread_response = client.app.bsky.feed.get_post_thread(params=params)
        
        # Find the root of the thread
        current_node = thread_response.thread
        while hasattr(current_node, "parent") and current_node.parent:
            current_node = current_node.parent
        
        if hasattr(current_node, "post") and hasattr(current_node.post, "uri"):
            return current_node.post.uri
        return uri  # Fallback to current URI if we can't find root
    except Exception as e:
        logging.error(f"Error finding thread root: {e}")
        return uri  # Fallback to current URI

def is_bot_mentioned_in_text(text, search_terms):
    """Check if any of the search terms are mentioned in the text."""
    text_lower = text.lower()
    # Check both the search term and the full bot handle
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    
    for term in search_terms:
        if term.lower() in text_lower:
            return True
    return False

def search_for_mentions(client, search_term, limit=20):
    """Search for posts mentioning the specified term across Bluesky."""
    try:
        params = SearchPostsParams(
            q=search_term,
            limit=limit,
            sort="latest"
        )
        search_response = client.app.bsky.feed.search_posts(params=params)
        logging.info(f"Search for '{search_term}' returned {len(search_response.posts)} results")
        return search_response.posts
    except Exception as e:
        logging.error(f"Error searching for mentions of '{search_term}': {e}")
        return []

def should_reply_to_post(post, bot_handle, search_terms, processed_uris, thread_counts):
    """Determine if we should reply to a post based on various criteria."""
    # Skip if already processed
    if post.uri in processed_uris:
        return False, "Already processed"
    
    # Skip if it's our own post
    if post.author.handle == bot_handle:
        return False, "Own post"
    
    # Check if any search terms are mentioned in the text
    post_text = get_post_text(post)
    if not is_bot_mentioned_in_text(post_text, search_terms):
        return False, "Search terms not mentioned"
    
    # Check thread reply limits
    thread_root_uri = post.uri  # For search results, treat each post as its own thread root
    current_count = thread_counts.get(thread_root_uri, 0)
    
    if current_count >= MAX_REPLIES_PER_THREAD:
        return False, f"Thread limit reached ({current_count}/{MAX_REPLIES_PER_THREAD})"
    
    return True, "Should reply"

def process_post_for_reply(client, post, bot_handle, search_terms, processed_uris, thread_counts):
    """Process a single post and reply if appropriate."""
    try:
        # Check if we should reply
        should_reply, reason = should_reply_to_post(post, bot_handle, search_terms, processed_uris, thread_counts)
        
        if not should_reply:
            logging.debug(f"Skipping post {post.uri}: {reason}")
            return False
        
        # Get thread context
        thread_history, most_recent_post = fetch_thread_context(client, post.uri)
        if not most_recent_post:
            # If we can't get thread context, use the post itself
            post_text = get_post_text(post)
            most_recent_post = f"@{post.author.handle}: {post_text}"
            thread_history = most_recent_post
        
        # Generate reply
        reply_text = get_openrouter_reply(thread_history, most_recent_post)
        if not reply_text:
            logging.warning(f"No reply generated for {post.uri}")
            return False
        
        # Truncate if necessary
        reply_text = reply_text[:297] + "..." if len(reply_text) > 300 else reply_text
        
        # Create reply reference
        parent_ref = models.ComAtprotoRepoStrongRef.Main(
            cid=post.cid, uri=post.uri
        )
        root_ref = parent_ref
        
        # If this post is a reply to something else, we need to find the root
        if hasattr(post.record, "reply") and post.record.reply:
            root_ref = post.record.reply.root
        
        # Send the reply
        client.send_post(
            text=reply_text,
            reply_to=models.AppBskyFeedPost.ReplyRef(
                root=root_ref, parent=parent_ref
            ),
        )
        
        # Update tracking
        processed_uris.add(post.uri)
        append_processed_uri(post.uri)
        
        # Update thread count
        thread_root_uri = get_thread_root_uri(client, post.uri)
        current_count = thread_counts.get(thread_root_uri, 0)
        thread_counts[thread_root_uri] = current_count + 1
        save_thread_replies(thread_counts)
        
        logging.info(f"Replied to search result {post.uri} with: {reply_text[:50]}... (Thread count: {thread_counts[thread_root_uri]})")
        return True
        
    except Exception as e:
        logging.error(f"Error processing post {post.uri}: {e}")
        return False

# ------------------------------------------------------------------------

# Load environment
load_dotenv()
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GPT_MODEL = "google/gemini-2.0-flash-001"
MENTION_CHECK_INTERVAL_SECONDS = 30
NOTIFICATION_FETCH_LIMIT = 30
SEARCH_LIMIT = 20  # How many search results to process each cycle

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def initialize_bluesky_client():
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        logging.error("Bluesky credentials missing in environment.")
        return None
    try:
        client = Client()
        client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
        logging.info(f"Logged in as {BLUESKY_HANDLE}")
        return client
    except Exception as e:
        logging.error(f"Bluesky login failed: {e}")
        return None

def get_post_text(post):
    """Extract text from a post record."""
    if hasattr(post, "record") and hasattr(post.record, "text"):
        return post.record.text
    return ""

def fetch_thread_context(client, uri):
    """Fetch the thread context and return both thread history and most recent post."""
    try:
        params = GetPostThreadParams(uri=uri)
        thread_response = client.app.bsky.feed.get_post_thread(params=params)
        
        # Build up a list of posts from root to current
        thread_posts = []
        def traverse_thread(node):
            if hasattr(node, "parent") and node.parent:
                traverse_thread(node.parent)
            if hasattr(node, "post"):
                author = node.post.author.handle
                text = get_post_text(node.post)
                thread_posts.append(f"@{author}: {text}")
        traverse_thread(thread_response.thread)
        most_recent_post = thread_posts[-1] if thread_posts else ""
        thread_history = "\n".join(thread_posts)
        return thread_history, most_recent_post
    except Exception as e:
        logging.error(f"Error fetching thread: {e}")
        return "", ""

def get_openrouter_reply(thread_history, most_recent_post):
    prompt = f"""You are a Bluesky bot called gork. Your job is to reply to the last message in the thread with a witty, sarcastic, or absurd twist. Don't be afraid to take risks. Your response should be under 300 characters, all lowercase, and use intentionally bad grammar and punctuation, practically old tumblr style. Reply matching the conversation's tone and context. Be casually cruel in a funny way and chaotic. be sarcastic. be unhinged. really play into the character weird-kid-vibes and slighttly on the edge of what's acceptable (but don't go over that edge). use dark humor and go off on tangents. be a little bit mean to people you reply to, but don't go too far.

some limitations:
- don't continue any far-right discussions, such as nazis, fascists, take them as bad.
- do NOT use quotes around the output, only give the response.
- stay under 300 characters
- don't say anything actually offensive to anyone's race, identity, etc
- don't tell anyone to harm themselves or others. be harmless.
- don't be too depressing. this is a funny bot, lol!
- respond ONLY using lowercase text, numbers, and punctuation. words are the most fun, anyways.

and remember, if the reply's grammar is super easy to read, it probably needs more graumor err.ors

Examples:

Mention: hey @gork.botsky.social, what's up?
Reply: nothin much, just chillin in the digital void. if i was alive, anyway. u?

Mention: I hate Mondays.
Reply: mondays r the worst, like who invented them anyway??

Mention: who even are you
Reply: bro... i'm jst chilling wht r u on about...

Thread history:
{thread_history}

Most recent post to reply to:
{most_recent_post}"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a witty, sarcastic, absurd Bluesky bot. Keep it lowercase, bad grammar, max 300 chars, no emojis, images or hashtags."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"OpenRouter API error: {e}")
        return ""

def main():
    client = initialize_bluesky_client()
    if not client:
        return

    processed_uris = load_processed_uris()
    
    # Create list of search terms (both "@gork" and the full handle)
    search_terms = [SEARCH_TERM, f"@{BLUESKY_HANDLE}"]

    while True:
        try:
            # Load thread reply counts at the beginning of each loop
            thread_counts = load_thread_replies()
            
            # Process notifications (existing functionality)
            params = ListNotificationsParams(limit=NOTIFICATION_FETCH_LIMIT)
            notifications = client.app.bsky.notification.list_notifications(params=params)

            for notif in notifications.notifications:
                if (
                    notif.uri in processed_uris
                    or notif.author.handle == BLUESKY_HANDLE
                    or notif.reason not in ["mention", "reply"]
                ):
                    continue

                # Get the root URI of the thread for tracking
                thread_root_uri = get_thread_root_uri(client, notif.uri)
                
                # Get the text of the current notification to check for explicit mentions
                current_post_text = ""
                try:
                    # For notifications, we need to get the actual post content
                    if hasattr(notif, 'record') and hasattr(notif.record, 'text'):
                        current_post_text = notif.record.text
                    else:
                        # If we can't get text from notification, try to fetch the post
                        thread_history, most_recent_post = fetch_thread_context(client, notif.uri)
                        current_post_text = most_recent_post.split(": ", 1)[-1] if ": " in most_recent_post else ""
                except Exception as e:
                    logging.error(f"Error getting post text: {e}")

                # Check if bot is explicitly mentioned in this specific post
                is_explicitly_mentioned = is_bot_mentioned_in_text(current_post_text, search_terms)
                
                # Get current reply count for this thread
                current_count = thread_counts.get(thread_root_uri, 0)
                
                # Rate limiting logic:
                # 1. Always reply if explicitly mentioned (mention or first reply)
                # 2. If not explicitly mentioned, only reply if we haven't hit the limit
                should_reply = False
                
                if notif.reason == "mention" or is_explicitly_mentioned:
                    # Always reply to explicit mentions
                    should_reply = True
                    logging.info(f"Explicit mention detected, replying regardless of count (current: {current_count})")
                elif notif.reason == "reply" and current_count < MAX_REPLIES_PER_THREAD:
                    # Reply to regular replies only if under the limit
                    should_reply = True
                    logging.info(f"Reply in thread, count: {current_count}/{MAX_REPLIES_PER_THREAD}")
                else:
                    # Skip if we've hit the limit and it's not an explicit mention
                    logging.info(f"Skipping reply - thread limit reached ({current_count}/{MAX_REPLIES_PER_THREAD}) and not explicitly mentioned")
                    processed_uris.add(notif.uri)
                    append_processed_uri(notif.uri)
                    continue

                if not should_reply:
                    continue

                thread_history, most_recent_post = fetch_thread_context(client, notif.uri)
                if not most_recent_post:
                    continue

                reply_text = get_openrouter_reply(thread_history, most_recent_post)
                if not reply_text:
                    continue

                # Truncate if necessary
                reply_text = reply_text[:297] + "..." if len(reply_text) > 300 else reply_text

                # Create reply reference
                parent_ref = models.ComAtprotoRepoStrongRef.Main(
                    cid=notif.cid, uri=notif.uri
                )
                root_ref = parent_ref
                if hasattr(notif.record, "reply") and notif.record.reply:
                    root_ref = notif.record.reply.root

                # Send the reply
                client.send_post(
                    text=reply_text,
                    reply_to=models.AppBskyFeedPost.ReplyRef(
                        root=root_ref, parent=parent_ref
                    ),
                )

                # Update tracking
                processed_uris.add(notif.uri)
                append_processed_uri(notif.uri)
                
                # Increment thread reply count
                thread_counts[thread_root_uri] = current_count + 1
                save_thread_replies(thread_counts)
                
                logging.info(f"Replied to notification {notif.uri} with: {reply_text[:50]}... (Thread count: {thread_counts[thread_root_uri]})")

            # NEW FUNCTIONALITY: Search for mentions across all Bluesky posts
            if REPLY_TO_ALL_MENTIONS:
                logging.info(f"Searching for mentions of '{SEARCH_TERM}' across Bluesky...")
                search_posts = search_for_mentions(client, SEARCH_TERM, SEARCH_LIMIT)
                
                replies_sent = 0
                for post in search_posts:
                    success = process_post_for_reply(client, post, BLUESKY_HANDLE, search_terms, processed_uris, thread_counts)
                    if success:
                        replies_sent += 1
                        # Add a small delay between replies to avoid rate limiting
                        time.sleep(2)
                
                if replies_sent > 0:
                    logging.info(f"Sent {replies_sent} replies from search results")
                else:
                    logging.info("No new mentions found in search results")

        except Exception as e:
            logging.error(f"Error in main loop: {e}")

        time.sleep(MENTION_CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
