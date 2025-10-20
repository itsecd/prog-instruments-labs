import aiohttp
import datetime


def prettify_user_info(user_info: dict) -> str:
    """Prettify user info

    Args:
        user_info (dict)

    Returns:
        str: result pretty string
    """
    # html injection ???   
    prettified = ""

    site_admin = user_info["site_admin"]
    if site_admin:
        prettified += "✨ADMIN\n"

    prettified += f"<b>✏️Login (username):</b> <code>{user_info['login']}</code>\n"

    name = user_info["name"]
    if name is not None:
        prettified += f"<b>🤵Name:</b> <code>{name}</code>\n"
    
    bio = user_info["bio"]
    if bio is not None:
        prettified += f"<b>📒Bio:</b>\n<blockquote>{bio}</blockquote>\n\n"
    
    company = user_info["company"]
    if company is not None:
        prettified += f"<b>💼Company:</b> <code>{company}</code>\n"
    
    email = user_info["email"]
    if email is not None:
        prettified += f"<b>📧Email:</n> {email}"
    
    created_date = datetime.datetime.strptime(user_info['created_at'], "%Y-%m-%dT%H:%M:%SZ")
    created_date_string = datetime.datetime.strftime(created_date, "%Y/%m/%d")
    prettified += f"<b>🕒Created at:</b> <code>{created_date_string}</code>\n\n"

    prettified += f"<a href=\"{user_info['url']}\">🔗Link</a>"

    return prettified


async def get_github_user(username: str) -> dict | None:
    """Fetch github user info with github public api

    Args:
        username (str)

    Raises:
        RuntimeError: sometring wrong with response

    Returns:
        dict | None: user info or nothing
    """
    url = f"https://api.github.com/users/{username}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 404:
                return None

            j = await response.json()
            if isinstance(j, dict) and j.get("login") == username:
                return j

    raise RuntimeError(f"Someting went wrong: {j}")
