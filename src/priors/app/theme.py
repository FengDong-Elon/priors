"""The look of the app: theme settings and a little CSS for Priors' own page elements.

The same theme is written to ``.streamlit/config.toml`` (read by Streamlit Community Cloud); ``priors app``
passes it on the command line, so it applies wherever the app is started. A test keeps the two in sync.
"""

from __future__ import annotations

NAVY = "#1F3A5F"
INK = "#1C2430"
MUTED = "#5B6B7F"
RULE = "#DDE2E8"
PAPER = "#F4F6F9"

THEME = {
    "base": "light",
    "primaryColor": NAVY,
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": PAPER,
    "textColor": INK,
    "linkColor": NAVY,
    "borderColor": RULE,
    "showWidgetBorder": True,
    "baseRadius": "small",
    "font": "sans-serif",
    "headingFont": "serif",
    "showSidebarBorder": True,
}
SIDEBAR = {"backgroundColor": "#F7F8FA"}
CLIENT = {"toolbarMode": "minimal"}   # no Deploy button or developer menu for students


def _flag(value) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def cli_flags() -> list[str]:
    """The theme as ``streamlit run`` options."""
    return ([f"--theme.{k}={_flag(v)}" for k, v in THEME.items()]
            + [f"--theme.sidebar.{k}={_flag(v)}" for k, v in SIDEBAR.items()]
            + [f"--client.{k}={_flag(v)}" for k, v in CLIENT.items()])


SERIF = "'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 4.2rem; }}
.pr-kicker {{ font-size: .72rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: {MUTED}; }}
.pr-mast {{ border-bottom: 1px solid {RULE}; padding-bottom: 1rem; margin-bottom: 1.3rem; }}
.pr-title {{ font-family: {SERIF}; font-size: 2.35rem; font-weight: 600; color: {INK}; line-height: 1.15;
            margin: .3rem 0 .4rem; }}
.pr-sub {{ color: #4A5565; font-size: 1.02rem; line-height: 1.55; max-width: 780px; }}
.pr-section {{ font-family: {SERIF}; font-size: 1.3rem; font-weight: 600; color: {INK}; margin: 1.4rem 0 .6rem; }}
.pr-steps {{ display: flex; flex-wrap: wrap; border: 1px solid {RULE}; border-radius: 6px; overflow: hidden;
            margin: .2rem 0 1.4rem; background: #fff; }}
.pr-step {{ flex: 1 1 130px; padding: .6rem .85rem; border-right: 1px solid {RULE}; font-size: .82rem;
           color: {MUTED}; line-height: 1.35; }}
.pr-step:last-child {{ border-right: none; }}
.pr-step .n {{ font-size: .68rem; letter-spacing: .09em; text-transform: uppercase; color: #8592A3; }}
.pr-step b {{ display: block; color: {INK}; font-size: .9rem; font-weight: 600; margin: .1rem 0; }}
.pr-step.done {{ background: {PAPER}; }}
.pr-step.done .n {{ color: #2E6B4F; }}
.pr-step.now {{ box-shadow: inset 0 -3px 0 {NAVY}; }}
.pr-step.now .n {{ color: {NAVY}; }}
.pr-brand {{ font-family: {SERIF}; font-size: 1.75rem; font-weight: 600; color: {INK}; line-height: 1.1; }}
.pr-brand-sub {{ font-size: .8rem; color: {MUTED}; margin: .15rem 0 .2rem; }}
.pr-label {{ font-size: .68rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: #8592A3;
            margin: .3rem 0 .35rem; }}
.pr-status {{ font-size: .88rem; color: {INK}; }}
.pr-dot {{ display: inline-block; width: .5rem; height: .5rem; border-radius: 50%; background: #2E8B57;
          margin-right: .45rem; vertical-align: middle; }}
.pr-kv {{ width: 100%; font-size: .84rem; border-collapse: collapse; }}
.pr-kv td {{ padding: .22rem 0; border-bottom: 1px solid {RULE}; vertical-align: top; }}
.pr-kv td:first-child {{ color: {MUTED}; width: 42%; padding-right: .5rem; }}
.pr-foot {{ font-size: .76rem; color: {MUTED}; line-height: 1.5; }}
.pr-foot a {{ color: {NAVY}; text-decoration: none; }}
[data-testid="stChatMessageAvatarAssistant"] {{ background-color: {NAVY}; color: #fff; }}
[data-testid="stChatMessageAvatarUser"] {{ background-color: #8FA3BC; color: #fff; }}
</style>
"""
