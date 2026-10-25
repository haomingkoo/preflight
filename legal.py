"""Static privacy and terms pages, written from what Preflight does in app.py."""
from __future__ import annotations

CONTACT = "[CONTACT EMAIL — owner to confirm]"
LAST_UPDATED = "25 October 2026"

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title} · Preflight</title>
<style>
@font-face {{ font-family: 'Inter'; font-weight: 100 900; font-display: swap;
  src: url('/assets/fonts/inter-latin-wght-normal.woff2') format('woff2'); }}
:root {{ color-scheme: dark; }}
body {{ margin: 0; background: #080808; color: #f0f0f0;
  font: 16px/1.65 'Inter', system-ui, sans-serif; }}
.legal {{ max-width: 68ch; margin: 0 auto; padding: 48px 20px 64px; }}
h1 {{ font-size: 2rem; line-height: 1.15; letter-spacing: -0.02em; margin: 24px 0 4px; }}
h2 {{ font-size: 1.15rem; margin: 2.2em 0 0.4em; }}
a {{ color: #7aa7ff; text-underline-offset: 3px; }}
a:focus-visible {{ outline: 2px solid #7aa7ff; outline-offset: 3px; border-radius: 2px; }}
.legal-meta, .legal-back, .legal-footer {{ color: #a1a1aa; font-size: 0.875rem; }}
.legal-footer {{ margin-top: 48px; }}
mark {{ background: #3f3a12; color: #fde68a; padding: 0 4px; border-radius: 3px; }}
li {{ margin: 0.35em 0; }}
</style>
</head>
<body>
<main class="legal">
<p class="legal-back"><a href="/">Back to Preflight</a></p>
<h1>{title}</h1>
<p class="legal-meta">Last updated {updated}</p>
{body}
<p class="legal-footer"><a href="/privacy">Privacy</a> · <a href="/terms">Terms</a></p>
</main>
</body>
</html>
"""

PRIVACY_BODY = f"""
<p>Preflight is a free tool that checks a dataset and trains a quick baseline model. It is run by Haoming Koo
as part of kooexperience.com. This page explains what happens to your file and to the small amount of
technical data the site needs. <mark>[LAWYER REVIEW]</mark></p>

<h2>What we collect</h2>
<ul>
<li><strong>The file you upload.</strong> Your browser sends the CSV or Parquet file to our server over HTTPS.
We read it into memory and save a compressed copy on the server's temporary disk under a random session ID.
We use it only to show you the analysis and models on this page.</li>
<li><strong>A session cookie.</strong> One strictly necessary cookie holds a random session ID so the server can find your
file. It expires after 8 hours. It is not used for tracking.</li>
<li><strong>Your IP address.</strong> We keep it in server memory for about a minute to rate-limit uploads and model
training. It is then discarded.</li>
</ul>
<p>We do not ask for your name or email. There are no accounts, advertising or session recording.
Fonts and styles are served from our own server, so loading the page does not contact Google Fonts or other CDNs.</p>
<p>The site uses <strong>Cloudflare Web Analytics</strong> (cookieless, aggregate page views, no input capture).
Cloudflare adds its script to the page at the network edge. It does not set cookies or read what you type or upload.</p>

<h2>How long we keep your file</h2>
<p>Your file is deleted 30 minutes after you upload it. A background task checks every minute. You can delete it
at any time with the <strong>Delete my data now</strong> button. Uploading a new file replaces the old one. Restarting
or redeploying the server also wipes all files.</p>

<h2>Who else handles your data</h2>
<p>We do not sell or share your file. We do not use it to train models outside your session. Two infrastructure
providers carry it on our behalf:</p>
<ul>
<li><strong>Cloudflare</strong> sits in front of the site as a secure proxy. The uploaded file passes through it in transit.
Cloudflare also provides the cookieless page-view analytics described above.</li>
<li><strong>Railway</strong> hosts the server and its temporary disk. Our traffic is served from Railway's Singapore edge.
<mark>[LAWYER REVIEW: confirm the Railway server region and both providers' data processing terms.]</mark></li>
</ul>
<p>Each provider may keep its own infrastructure logs under its own policy.</p>

<h2>Logs</h2>
<p>The app does not keep access logs. When something goes wrong it logs the type of error and a random request ID.
It never logs error messages or tracebacks, because those can contain values from your file.</p>

<h2>Personal data inside your file</h2>
<p>If your file contains personal data about other people, you are responsible for having the right to process it.
Preflight only processes that data on your instructions, for the analysis you ask for. Please remove or anonymise
personal data you do not need. <mark>[LAWYER REVIEW: confirm the data intermediary / processor position under the
PDPA and GDPR.]</mark></p>

<h2>Children</h2>
<p>Preflight is not directed at children. Do not upload data about children unless you have the authority to do so.</p>

<h2>Your rights</h2>
<p>You can ask what personal data we hold about you or ask us to correct or delete it. In practice we hold almost
nothing once your file has expired. If you are in the EU or UK you also have rights under the GDPR, including the
right to complain to your data protection authority. In Singapore you can contact the PDPC. <mark>[LAWYER REVIEW]</mark></p>

<h2>Contact</h2>
<p>Questions or requests: {CONTACT}</p>

<h2>Changes</h2>
<p>If we change how Preflight handles data, we will update this page and the date above.</p>
"""

TERMS_BODY = f"""
<p>These terms apply when you use Preflight. By uploading a file you agree to them. <mark>[LAWYER REVIEW]</mark></p>

<h2>What Preflight is</h2>
<p>Preflight is a free tool for quick data checks and baseline models. Results are automated and can be wrong.
They are for exploration and teaching. Do not rely on them for decisions about people, money, health or safety
without your own review.</p>

<h2>Who can use it</h2>
<p>You must be 18 or older. Students aged 13 to 17 may use it only under the supervision of a teacher or school.
<mark>[LAWYER REVIEW: confirm the age threshold.]</mark></p>

<h2>Your promise about uploaded data</h2>
<p>When you upload a file you confirm that:</p>
<ul>
<li>you have the right to upload and process the data in it;</li>
<li>it contains no personal data about other people, or you have their permission or another lawful basis to process it;</li>
<li>it contains no data about children unless you have the authority to share it;</li>
<li>it contains no sensitive data such as health or financial records unless you are allowed to process it this way.</li>
</ul>
<p>You are responsible for the data you upload. You agree to cover any loss we suffer because you broke this promise.
<mark>[LAWYER REVIEW: indemnity wording.]</mark></p>

<h2>Acceptable use</h2>
<p>Do not try to get around upload or rate limits, or break or overload the service. Do not access other users' data.
Do not upload malware or content that is illegal to hold. We may block access to stop misuse.</p>

<h2>Your data</h2>
<p>You keep all rights in your data. We use it only to run the analysis you ask for. It is deleted as described in the
<a href="/privacy">privacy notice</a>.</p>

<h2>No warranty</h2>
<p>Preflight is provided as is. It may be unavailable, change, or stop at any time. To the extent the law allows, we
are not liable for any loss from using it or relying on its results. <mark>[LAWYER REVIEW: limitation of liability.]</mark></p>

<h2>Governing law</h2>
<p>These terms are governed by the laws of Singapore. <mark>[LAWYER REVIEW]</mark></p>

<h2>Contact</h2>
<p>{CONTACT}</p>
"""


def render(title: str, body: str) -> str:
    return _PAGE.format(title=title, updated=LAST_UPDATED, body=body)


PRIVACY_HTML = render("Privacy notice", PRIVACY_BODY)
TERMS_HTML = render("Terms of use", TERMS_BODY)
