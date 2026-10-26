"""Static privacy and terms pages, written from what Preflight does in app.py."""
from __future__ import annotations

CONTACT = '<a href="mailto:privacy@kooexperience.com">privacy@kooexperience.com</a>'
LAST_UPDATED = "25 October 2026"

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}: Preflight</title>
<link rel="icon" type="image/svg+xml" href="/assets/favicon.svg" />
<style>
@font-face {{ font-family: 'Hanken Grotesk'; font-weight: 100 900; font-display: swap;
  src: url('/assets/fonts/hanken-grotesk-latin-wght-normal.woff2') format('woff2'); }}
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #f4f6f6; color: #141a1d;
  font: 1rem/1.65 'Hanken Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
::selection {{ background: #c7e5e1; color: #141a1d; }}
.wrap {{ max-width: 1200px; margin: 0 auto; padding: 0 1rem; }}
header {{ border-bottom: 1px solid #dbe1e1; }}
header .wrap {{ display: flex; align-items: center; justify-content: space-between; min-height: 76px; }}
.logo {{ display: inline-flex; align-items: center; gap: .75rem; min-height: 44px; color: #141a1d; font-weight: 700; text-decoration: none; }}
header nav a {{ display: inline-flex; align-items: center; min-height: 44px; color: #3a4448; font-weight: 500; text-decoration: none; }}
.legal {{ max-width: 68ch; padding: 3rem 0 0; margin: 0 1rem 0 max(1rem, calc((100% - 1200px) / 2 + 1rem)); }}
h1 {{ font-size: 2rem; line-height: 1.15; letter-spacing: -0.02em; font-weight: 650; margin: 0 0 0.25rem; }}
h2 {{ font-size: 1.125rem; line-height: 1.3; font-weight: 600; margin: 2em 0 0.5em; }}
p, li {{ color: #3a4448; text-wrap: pretty; }}
strong {{ color: #141a1d; font-weight: 600; }}
a {{ color: #0f766e; text-underline-offset: 3px; }}
a:hover {{ color: #0b5953; }}
a:focus-visible {{ outline: 2px solid #0f766e; outline-offset: 3px; border-radius: 2px; }}
.legal-meta {{ color: #56616a; font-size: 0.875rem; margin: 0; }}
li {{ margin: 0.4em 0; }}
footer {{ margin-top: 4rem; border-top: 1px solid #dbe1e1; }}
footer .wrap {{ padding: 1.5rem 1rem 2rem; font-size: 0.8125rem; color: #56616a; display: flex; flex-wrap: wrap; }}
footer a {{ color: #56616a; }}
footer .sep {{ padding: 0 0.5rem; }}
</style>
</head>
<body>
<header><div class="wrap">
<a class="logo" href="/"><img src="/assets/favicon.svg" alt="" width="24" height="24" />Preflight</a>
<nav aria-label="Main"><a href="https://github.com/haomingkoo/preflight" rel="noopener">GitHub</a></nav>
</div></header>
<main class="legal">
<h1>{title}</h1>
{meta}
{body}
</main>
<footer><div class="wrap">
<span>&copy; 2026 Haoming Koo</span><span class="sep" aria-hidden="true">&middot;</span>
<a href="/privacy">Privacy</a><span class="sep" aria-hidden="true">&middot;</span>
<a href="/terms">Terms</a><span class="sep" aria-hidden="true">&middot;</span>
<span>Contact ({contact})</span><span class="sep" aria-hidden="true">&middot;</span>
<span>More projects &rarr; <a href="https://kooexperience.com/">kooexperience.com</a></span>
</div></footer>
</body>
</html>
"""

PRIVACY_BODY = f"""
<p>Preflight is a free tool for checking a dataset and training a quick baseline model.
Haoming Koo runs it as part of kooexperience.com. This page covers what happens to your file
and the little technical data the site needs.</p>

<h2>What we collect</h2>
<ul>
<li><strong>Your file.</strong> Your browser sends it to our server over HTTPS. We read it into memory
and keep a compressed copy on the server's temporary disk under a random session ID. We use it only to
show you the analysis on this page.</li>
<li><strong>A session cookie.</strong> It holds the random session ID so the server can find your file.
It expires after 8 hours and is not used for tracking.</li>
<li><strong>Your IP address.</strong> We hold it in server memory for about a minute to rate-limit uploads
and model training.</li>
</ul>
<p>We do not ask for your name or email. There are no accounts or ads, and nothing records your session.
Fonts and styles come from our own server.</p>
<p>The site uses Cloudflare Web Analytics (cookieless, aggregate page views, no input capture).
Cloudflare adds its script at the network edge. It sets no cookies and does not see what you type or upload.</p>

<h2>How long we keep your file</h2>
<p>We delete your file 30 minutes after you upload it. A background task checks every minute.
You can delete it sooner with the <strong>Delete my data now</strong> button. A new upload replaces the
old file. A server restart also removes every file.</p>

<h2>Who else handles your data</h2>
<p>We do not sell or share your file. Two providers carry it for us:</p>
<ul>
<li><strong>Cloudflare</strong> sits in front of the site as a secure proxy. Your file passes through it on the way
to our server. Cloudflare also runs the page-view analytics above.</li>
<li><strong>Railway</strong> hosts the server and its temporary disk. Traffic is served from Railway's Singapore edge.</li>
</ul>
<p>Each provider keeps its own infrastructure logs under its own policy.</p>

<h2>Logs</h2>
<p>The app keeps no access logs. When something fails it records the type of error and a random request ID.
Error messages can contain values from your file, so the app never logs them.</p>

<h2>Personal data in your file</h2>
<p>If your file holds personal data about other people, you need the right to process it. Preflight processes
it only to run the analysis you ask for. Remove or anonymise any personal data you do not need.</p>

<h2>Children</h2>
<p>Preflight is not meant for children. Only upload data about children if you have the authority to do so.</p>

<h2>Your rights</h2>
<p>You can ask what personal data we hold about you. You can also ask us to correct or delete it.
Once your file expires we hold almost nothing. In the EU or UK you have rights under the GDPR, including the
right to complain to your data protection authority. In Singapore you can contact the PDPC.</p>

<h2>Contact</h2>
<p>Email {CONTACT} with any question or request.</p>

<h2>Changes</h2>
<p>If we change how Preflight handles data, we will update this page and the date above.</p>
"""

TERMS_BODY = f"""
<p>These terms apply when you use Preflight. Uploading a file means you accept them.</p>

<h2>What Preflight is</h2>
<p>Preflight is a free tool for quick data checks and baseline models. Its results are automated and can be wrong.
Use them for exploration and teaching. Review them yourself before using them in decisions about people or money.
The same applies to health and safety.</p>

<h2>Who can use it</h2>
<p>You must be 18 or older. Students aged 13 to 17 may use it with a teacher or school supervising.</p>

<h2>What you promise about your data</h2>
<p>When you upload a file you confirm that:</p>
<ul>
<li>you have the right to upload and process the data in it;</li>
<li>it holds no personal data about other people, or you have their permission or another lawful basis;</li>
<li>it holds no data about children unless you have the authority to share it;</li>
<li>any sensitive data in it, such as health or financial records, is yours to process this way.</li>
</ul>
<p>You are responsible for the data you upload. If breaking these promises causes us a loss, you agree to cover it.</p>

<h2>Acceptable use</h2>
<p>Do not try to get around the upload or rate limits. Do not try to break or overload the service, or reach other
people's data. Do not upload malware or anything illegal to hold. We may block access to stop misuse.</p>

<h2>Your data</h2>
<p>Your data stays yours. We use it only to run the analysis you ask for and delete it as the
<a href="/privacy">privacy notice</a> describes.</p>

<h2>No warranty</h2>
<p>Preflight is provided as is. It may be unavailable or change at any time, and it may stop.
As far as the law allows, we are not liable for losses from using it or relying on its results.</p>

<h2>Governing law</h2>
<p>The laws of Singapore govern these terms.</p>

<h2>Contact</h2>
<p>{CONTACT}</p>
"""


def render(title: str, body: str, meta: bool = True) -> str:
    line = f'<p class="legal-meta">Last updated {LAST_UPDATED}</p>' if meta else ""
    return _PAGE.format(title=title, meta=line, body=body, contact=CONTACT)


PRIVACY_HTML = render("Privacy notice", PRIVACY_BODY)
TERMS_HTML = render("Terms of use", TERMS_BODY)
NOT_FOUND_HTML = render(
    "Page not found",
    '<p>There is nothing at this address.</p>\n'
    '<p><a href="/">Back to Preflight</a> or see more projects at '
    '<a href="https://kooexperience.com/">kooexperience.com</a>.</p>',
    meta=False,
)
