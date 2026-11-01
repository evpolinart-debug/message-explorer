# ============================================================
# 💬 MESSAGE EXPLORER — анализ сообщений из CSV  (Google Colab)
# Вставь весь код в одну ячейку и запусти (Shift+Enter)
# ============================================================
import os, re, io, csv, base64, random, datetime, warnings, subprocess, sys, html as html_lib
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Markdown

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 120)

FILE_PATH = os.environ.get("MSG_FILE", "/content/Messages_МосУ_МВД_России_имени_В_Я_Кикотя.csv")

# Если автоопределение колонок ошиблось — впиши названия вручную, например TEXT_COL = "text"
TEXT_COL = None
DATE_COL = None
AUTHOR_COL = None

try:
    import ipywidgets as widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
WEEKDAYS_FULL = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
PALETTE = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6"]
FIGS = []  # графики для HTML-отчёта


# ---------- 🛠 Оформление ----------
def banner(text, emoji="✨", c1="#4F46E5", c2="#0EA5E9"):
    display(HTML(f"""
    <div style="background:linear-gradient(90deg,{c1},{c2});color:white;
                padding:12px 20px;border-radius:12px;margin:22px 0 8px 0;
                font-size:20px;font-weight:bold;font-family:sans-serif;
                box-shadow:0 4px 12px rgba(0,0,0,.18)">{emoji} {text}</div>"""))


def stat_cards(items):
    cards = "".join(
        f"""<div style="flex:1;min-width:140px;background:white;border-radius:12px;padding:14px;
                        box-shadow:0 2px 8px rgba(0,0,0,.12);text-align:center;color:#222">
              <div style="font-size:26px">{e}</div>
              <div style="font-size:22px;font-weight:bold">{v}</div>
              <div style="font-size:12px;color:#666">{l}</div></div>"""
        for e, v, l in items)
    display(HTML(f'<div style="display:flex;flex-wrap:wrap;gap:12px;font-family:sans-serif">{cards}</div>'))


def show(title):
    """Показать текущий график и сохранить его для HTML-отчёта."""
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    FIGS.append((title, base64.b64encode(buf.getvalue()).decode()))
    plt.show()


def fmt(n):
    return f"{n:,}".replace(",", " ")


# ---------- 📂 Умная загрузка CSV ----------
def read_csv_smart(path):
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            with open(path, encoding=enc) as f:
                sample = f.read(50000)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        try:
            first = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            first = ","
        for sep in dict.fromkeys([first, ",", ";", "\t", "|"]):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, on_bad_lines="skip", low_memory=False)
            except Exception as e:
                last_err = e
                continue
            if df.shape[1] > 1 or sep == "|":
                return df, enc, sep
    raise RuntimeError(f"Не удалось прочитать файл: {last_err}")


# ---------- 🧭 Автоопределение колонок ----------
TEXT_KEYS = ["text", "message", "msg", "content", "body", "post", "сообщ", "текст", "содерж", "запис"]
DATE_KEYS = ["date", "time", "created", "timestamp", "дата", "время"]
AUTHOR_KEYS = ["from", "author", "sender", "user", "name", "nick", "автор", "отправ", "пользов", "имя", "от кого"]
ENG_KEYS = ["view", "like", "reaction", "forward", "repost", "comment", "просмотр", "лайк", "реакц", "репост", "коммент"]


def _is_str(s):
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and not pd.api.types.is_numeric_dtype(s)


def _named(df, keys):
    return [c for c in df.columns if any(k in str(c).lower() for k in keys)]


def parse_dates(s):
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if len(v) and v.between(1e9, 2.5e9).mean() > .8:
            return pd.to_datetime(s, unit="s", errors="coerce")
        if len(v) and v.between(1e12, 2.5e12).mean() > .8:
            return pd.to_datetime(s, unit="ms", errors="coerce")
        return None
    best = None
    for dayfirst in (False, True):
        try:
            p = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, utc=True).dt.tz_convert(None)
        except Exception:
            continue
        if best is None or p.notna().mean() > best.notna().mean():
            best = p
    return best


def detect_date(df):
    if DATE_COL:
        return DATE_COL, parse_dates(df[DATE_COL])
    named = _named(df, DATE_KEYS)
    others = [c for c in df.columns if c not in named and _is_str(df[c])
              and df[c].dropna().astype(str).str.len().mean() < 40]
    for c in named + others:
        p = parse_dates(df[c])
        if p is not None and p.notna().mean() > .6 and p.dt.year.between(1990, 2100).mean() > .6:
            return c, p
    return None, None


def detect_text(df, exclude):
    if TEXT_COL:
        return TEXT_COL
    obj = [c for c in df.columns if _is_str(df[c]) and c not in exclude]
    named = [c for c in _named(df, TEXT_KEYS) if c in obj]
    pool = named or obj
    if not pool:
        return None
    return max(pool, key=lambda c: df[c].dropna().astype(str).str.len().mean() if df[c].notna().any() else 0)


def detect_author(df, exclude):
    if AUTHOR_COL:
        return AUTHOR_COL
    n = len(df)
    def ok(c):
        s = df[c].dropna().astype(str)
        return len(s) and 1 <= s.nunique() <= max(50, n * .5) and s.str.len().mean() < 60
    named = [c for c in _named(df, AUTHOR_KEYS) if c not in exclude and _is_str(df[c]) and ok(c)]
    if named:
        return named[0]
    pool = [c for c in df.columns if c not in exclude and _is_str(df[c]) and ok(c)
            and not any(k in str(c).lower() for k in ["type", "тип", "kind", "id"])
            and 2 <= df[c].nunique() <= min(500, max(2, n * .3))]
    return pool[0] if pool else None


RAW, ENCODING, SEP = read_csv_smart(FILE_PATH)
RAW.columns = [str(c).strip() for c in RAW.columns]
D_COL, DATES = detect_date(RAW)
T_COL = detect_text(RAW, {D_COL})
A_COL = detect_author(RAW, {D_COL, T_COL})
ENG_COLS = [c for c in _named(RAW, ENG_KEYS)
            if pd.api.types.is_numeric_dtype(RAW[c]) and c not in (T_COL, D_COL, A_COL)]

if T_COL is None:
    raise ValueError("Не нашёл колонку с текстом сообщений. Укажи её вручную: TEXT_COL = 'название'")

M = pd.DataFrame({"text": RAW[T_COL].fillna("").astype(str).str.strip()})
M["date"] = DATES if DATES is not None else pd.NaT
M["author"] = RAW[A_COL].fillna("—").astype(str) if A_COL else "—"
for c in ENG_COLS:
    M[c] = RAW[c]
EMPTY_TEXTS = int((M["text"] == "").sum())
M = M[M["text"] != ""].reset_index(drop=True)
if DATES is not None:
    M = M.sort_values("date", na_position="last").reset_index(drop=True)

HAS_DATE = M["date"].notna().any()
HAS_TIME = HAS_DATE and bool((M["date"].dt.hour + M["date"].dt.minute).fillna(0).sum() > 0)
HAS_AUTHOR = A_COL is not None and M["author"].nunique() > 1


# ---------- 🔤 Слова ----------
STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще
нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом
себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет
ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем
всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве
три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
это также этих свой своих своей наш наши нашего нашей нашем вас ваш ваши год года году лет также который которые
которых которая которое которой которого которым также очень весь всё все всем всеми тех том тому таким такие такая
http https www com ru the and for you are with this that from have was not but all
""".replace("ё", "е").split())
WORD_RE = re.compile(r"[a-zа-яё]+(?:-[a-zа-яё]+)?", re.I)
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")
LINK_RE = re.compile(r"https?://(?:www\.)?([^/\s]+)", re.I)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


def tokens(text):
    text = LINK_RE.sub(" ", text.lower().replace("ё", "е"))
    return [w for w in WORD_RE.findall(text) if len(w) > 2 and w not in STOP]


M["tokens"] = M["text"].map(tokens)
M["length"] = M["text"].str.len()
M["n_words"] = M["text"].str.split().str.len()
WORDS = Counter(w for ts in M["tokens"] for w in ts)


# ============================================================
# 0. 🎉 Заставка
# ============================================================
display(HTML(f"""
<div style="text-align:center;padding:26px;border-radius:18px;
            background:linear-gradient(135deg,#1E3A8A,#4F46E5,#0EA5E9);color:white;font-family:sans-serif">
  <div style="font-size:34px;font-weight:bold">💬 Message Explorer</div>
  <div style="font-size:15px;margin-top:6px;opacity:.9">{html_lib.escape(os.path.basename(FILE_PATH))}</div>
  <div style="font-size:13px;margin-top:4px;opacity:.75">кодировка {ENCODING} · разделитель «{SEP}» ·
  {datetime.datetime.now():%d.%m.%Y %H:%M}</div>
</div>"""))

period_days = (M["date"].max() - M["date"].min()).days + 1 if HAS_DATE else None
stat_cards([
    ("💬", fmt(len(M)), "сообщений с текстом"),
    ("👥", fmt(M["author"].nunique()) if HAS_AUTHOR else "—", "авторов"),
    ("📅", fmt(period_days) if period_days else "—", "дней в периоде"),
    ("🔤", fmt(int(M["n_words"].sum())), "слов всего"),
    ("📏", fmt(int(M["length"].median())), "символов (медиана)"),
    ("🧠", fmt(len(WORDS)), "уникальных слов"),
])
if HAS_DATE:
    print(f"Период: {M['date'].min():%d.%m.%Y} — {M['date'].max():%d.%m.%Y}")


# ============================================================
# 1. 🧭 Структура файла
# ============================================================
banner("Структура файла и найденные колонки", "🧭")
display(Markdown(
    f"| Роль | Колонка |\n|---|---|\n"
    f"| 💬 Текст | `{T_COL}` |\n| 📅 Дата | `{D_COL or 'не найдена'}` |\n"
    f"| 👤 Автор | `{A_COL or 'не найден'}` |\n| 📊 Метрики | `{', '.join(ENG_COLS) or 'нет'}` |"))
info = pd.DataFrame({
    "Колонка": RAW.columns,
    "Тип": [str(t) for t in RAW.dtypes],
    "Заполнено, %": (RAW.notna().mean() * 100).round(1).values,
    "Уникальных": [RAW[c].nunique() for c in RAW.columns],
    "Пример": [str(RAW[c].dropna().iloc[0])[:60] if RAW[c].notna().any() else "" for c in RAW.columns],
})
display(info.style.bar(subset=["Заполнено, %"], color="#A5B4FC", vmin=0, vmax=100).hide(axis="index"))
dups = int(M["text"].duplicated().sum())
print(f"🩺 Строк в файле: {fmt(len(RAW))} · пустых текстов: {fmt(EMPTY_TEXTS)} · "
      f"повторяющихся сообщений: {fmt(dups)}")


# ============================================================
# 2. 📈 Активность во времени
# ============================================================
if HAS_DATE:
    banner("Активность во времени", "📈", "#0EA5E9", "#10B981")
    daily = M.set_index("date").resample("D").size()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily.index, daily.values, color="#93C5FD", alpha=.5, label="в день")
    ax.plot(daily.rolling(7, min_periods=1).mean(), color="#1D4ED8", lw=2, label="среднее за 7 дней")
    ax.set_title("Сообщения по дням"); ax.set_ylabel("сообщений"); ax.legend()
    show("Сообщения по дням")

    monthly = M.set_index("date").resample("MS").size()
    if len(monthly) > 1:
        fig, ax = plt.subplots(figsize=(13, 3.6))
        colors = ["#F59E0B" if v == monthly.max() else "#6366F1" for v in monthly.values]
        ax.bar(monthly.index.strftime("%m.%Y"), monthly.values, color=colors)
        ax.set_title("Сообщения по месяцам (оранжевый — рекорд)")
        plt.xticks(rotation=60, fontsize=8)
        show("Сообщения по месяцам")

    if HAS_TIME:
        fig = plt.figure(figsize=(14, 4.8))
        ax1 = fig.add_subplot(1, 2, 1)
        pivot = (M.assign(wd=M["date"].dt.weekday, h=M["date"].dt.hour)
                   .pivot_table(index="wd", columns="h", values="text", aggfunc="count")
                   .reindex(index=range(7), columns=range(24)).fillna(0))
        sns.heatmap(pivot, cmap="YlGnBu", ax=ax1, cbar_kws={"label": "сообщений"})
        ax1.set_yticklabels(WEEKDAYS, rotation=0); ax1.set_xlabel("час"); ax1.set_ylabel("")
        ax1.set_title("Тепловая карта: день недели × час")

        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        hours = M["date"].dt.hour.value_counts().reindex(range(24), fill_value=0)
        theta = np.arange(24) / 24 * 2 * np.pi
        ax2.bar(theta, hours.values, width=2 * np.pi / 24 * .9,
                color=plt.cm.plasma(hours.values / max(hours.max(), 1)))
        ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
        ax2.set_xticks(theta); ax2.set_xticklabels(range(24), fontsize=8); ax2.set_yticklabels([])
        ax2.set_title("Часы активности", pad=15)
        plt.tight_layout()
        show("Активность по дням недели и часам")


# ============================================================
# 3. 🏆 Рекорды
# ============================================================
banner("Рекорды", "🏆", "#F59E0B", "#EF4444")
records = []
if HAS_DATE:
    daily_nz = daily[daily > 0]
    records.append(("🔥 Самый активный день", f"{daily.idxmax():%d.%m.%Y} — {fmt(int(daily.max()))} сообщ."))
    wd = M["date"].dt.weekday.value_counts().idxmax()
    records.append(("📆 Любимый день недели", WEEKDAYS_FULL[wd]))
    if HAS_TIME:
        hr = int(M["date"].dt.hour.value_counts().idxmax())
        records.append(("⏰ Самый активный час", f"{hr:02d}:00–{hr:02d}:59"))
    days = pd.Series(sorted(daily_nz.index))
    gaps = days.diff().dt.days
    best, cur, best_end = 1, 1, days.iloc[0]
    for i in range(1, len(days)):
        cur = cur + 1 if gaps.iloc[i] == 1 else 1
        if cur > best:
            best, best_end = cur, days.iloc[i]
    records.append(("⛓ Самая длинная серия без пропусков",
                    f"{best} дн. подряд (до {best_end:%d.%m.%Y})"))
    if len(days) > 1 and gaps.max() > 1:
        gi = gaps.idxmax()
        records.append(("🌙 Самое долгое затишье",
                        f"{int(gaps.max())} дн. ({days.iloc[gi - 1]:%d.%m.%Y} → {days.iloc[gi]:%d.%m.%Y})"))
longest = M.loc[M["length"].idxmax()]
records.append(("📜 Самое длинное сообщение", f"{fmt(int(longest['length']))} символов"))
if HAS_AUTHOR:
    top_a = M["author"].value_counts()
    records.append(("👑 Самый активный автор", f"{top_a.index[0]} — {fmt(int(top_a.iloc[0]))} сообщ."))
if WORDS:
    w, c = WORDS.most_common(1)[0]
    records.append(("🔤 Самое частое слово", f"«{w}» — {fmt(c)} раз"))
display(HTML("<table style='font-family:sans-serif;font-size:15px'>" + "".join(
    f"<tr><td style='padding:6px 18px 6px 0'><b>{html_lib.escape(k)}</b></td>"
    f"<td>{html_lib.escape(v)}</td></tr>" for k, v in records) + "</table>"))


# ============================================================
# 4. 👥 Авторы
# ============================================================
if HAS_AUTHOR:
    banner("Авторы", "👥", "#8B5CF6", "#EC4899")
    top = M["author"].value_counts().head(15)[::-1]
    fig, ax = plt.subplots(figsize=(11, max(3, len(top) * .38)))
    ax.barh([a[:40] for a in top.index], top.values, color=sns.color_palette("magma_r", len(top)))
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_title("Топ-15 авторов по числу сообщений")
    show("Топ авторов")
    agg = {"Сообщений": ("text", "size"), "Средняя длина": ("length", "mean")}
    if HAS_DATE:
        agg.update({"Первое": ("date", "min"), "Последнее": ("date", "max")})
    authors_table = M.groupby("author").agg(**agg).sort_values("Сообщений", ascending=False).head(20)
    authors_table["Средняя длина"] = authors_table["Средняя длина"].round(0)
    display(authors_table.style.background_gradient(subset=["Сообщений"], cmap="Purples"))


# ============================================================
# 5. 📏 Длина сообщений
# ============================================================
banner("Длина сообщений", "📏", "#14B8A6", "#0EA5E9")
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
sns.histplot(M["length"].clip(lower=1), bins=40, log_scale=True, color="#14B8A6", ax=axes[0])
axes[0].set_title("Распределение длины (символы, лог. шкала)")
bins = [0, 20, 50, 100, 300, 1000, np.inf]
names = ["≤20", "21–50", "51–100", "101–300", "301–1000", ">1000"]
cats = pd.cut(M["length"], bins=bins, labels=names).value_counts().reindex(names)
axes[1].bar(names, cats.values, color=PALETTE[:6])
axes[1].set_title("Короткие, средние и «лонгриды»")
plt.tight_layout()
show("Длина сообщений")


# ============================================================
# 6. 🔤 Слова: топ, облако, пары слов
# ============================================================
banner("О чём пишут: слова и облако слов", "🔤", "#4F46E5", "#8B5CF6")
if WORDS:
    top_words = pd.Series(dict(WORDS.most_common(30)))[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_words.index, top_words.values, color=sns.color_palette("viridis", len(top_words)))
    ax.set_title("30 самых частых слов (без служебных)")
    show("Топ слов")

    try:
        from wordcloud import WordCloud
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wordcloud"],
                           check=True, capture_output=True, timeout=180)
            from wordcloud import WordCloud
        except Exception:
            WordCloud = None
    if WordCloud:
        wc = WordCloud(width=1400, height=650, background_color="white", colormap="viridis",
                       max_words=150).generate_from_frequencies(dict(WORDS.most_common(300)))
        plt.figure(figsize=(14, 6.5)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
        plt.title("Облако слов")
        show("Облако слов")

    bigrams = Counter(f"{a} {b}" for ts in M["tokens"] for a, b in zip(ts, ts[1:]))
    if bigrams:
        bg = pd.Series(dict(bigrams.most_common(15)))[::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(bg.index, bg.values, color="#F59E0B")
        ax.set_title("Частые пары слов")
        show("Пары слов")


# ============================================================
# 7. #️⃣ Хэштеги, упоминания, ссылки, эмодзи
# ============================================================
banner("Хэштеги, упоминания, ссылки и эмодзи", "#️⃣", "#EC4899", "#F59E0B")
extras = {
    "Хэштеги": Counter(h.lower() for t in M["text"] for h in HASHTAG_RE.findall(t)),
    "Упоминания": Counter(m.lower() for t in M["text"] for m in MENTION_RE.findall(t)),
    "Сайты в ссылках": Counter(d.lower() for t in M["text"] for d in LINK_RE.findall(t)),
}
EMOJIS = Counter(e for t in M["text"] for e in EMOJI_RE.findall(t))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (title, cnt), color in zip(axes.flat, extras.items(), PALETTE[3:6]):
    ax.set_title(f"{title} (всего: {fmt(sum(cnt.values()))})")
    if not cnt:
        ax.text(.5, .5, "не найдено", ha="center", va="center", fontsize=13, color="#999")
        ax.axis("off")
        continue
    s = pd.Series(dict(cnt.most_common(10)))[::-1]
    ax.barh([str(i)[:30] for i in s.index], s.values, color=color)
plt.tight_layout()
show("Хэштеги, упоминания, ссылки")
emoji_top = EMOJIS.most_common(15)
if emoji_top:
    display(HTML(f"<p style='font-family:sans-serif'><b>😀 Топ эмодзи</b> (всего: {fmt(sum(EMOJIS.values()))})</p>"
                 "<div style='font-size:26px'>" + " ".join(
        f"{e}<sub style='font-size:12px'>{n}</sub>" for e, n in emoji_top) + "</div>"))


# ============================================================
# 8. 🧩 Автоматические темы (NMF)
# ============================================================
TOPICS = []
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    docs = M.loc[M["length"] > 20, "tokens"].map(" ".join)
    if len(docs) >= 30:
        vec = TfidfVectorizer(max_df=.5, min_df=2, max_features=5000, token_pattern=r"\S+")
        X = vec.fit_transform(docs)
        k = int(min(6, max(2, len(docs) // 60)))
        if X.shape[1] > k * 3:
            banner("Автоматические темы сообщений", "🧩", "#10B981", "#14B8A6")
            nmf = NMF(n_components=k, random_state=0, init="nndsvda", max_iter=500)
            W = nmf.fit_transform(X)
            terms = vec.get_feature_names_out()
            share = np.bincount(W.argmax(axis=1), minlength=k) / len(docs) * 100
            for i, comp in enumerate(nmf.components_):
                words = ", ".join(terms[comp.argsort()[::-1][:8]])
                TOPICS.append((i + 1, round(float(share[i]), 1), words))
            display(pd.DataFrame(TOPICS, columns=["Тема", "Доля сообщений, %", "Ключевые слова"])
                    .style.bar(subset=["Доля сообщений, %"], color="#6EE7B7").hide(axis="index"))
            print("Темы найдены алгоритмом NMF по частоте слов: это подсказка, а не точная классификация.")
except Exception as e:
    print(f"Темы не построены: {e}")


# ============================================================
# 9. 🌡 Настроение (упрощённый словарный подход)
# ============================================================
POS = ("спасиб", "благодар", "поздрав", "отличн", "хорош", "рад", "успех", "побед", "лучш", "молодц",
       "прекрасн", "замечат", "любл", "любим", "счаст", "горд", "поддерж", "дружн", "ура", "здоров",
       "добр", "крут", "класс", "супер", "интересн", "вдохнов", "достиж", "награ", "почет", "уважен")
NEG = ("плох", "проблем", "ошибк", "жаль", "груст", "ужас", "отмен", "опасн", "трудн", "конфликт",
       "авари", "скорб", "погиб", "утрат", "печаль", "злой", "обид", "провал", "неудач", "беда",
       "бол", "страх", "тревог", "жалоб", "недовол", "сожален", "кошмар", "увы")
vocab = {w: (1 if w.startswith(POS) else -1 if w.startswith(NEG) else 0) for w in WORDS}
M["mood"] = M["tokens"].map(lambda ts: sum(vocab.get(w, 0) for w in ts))
M["mood_label"] = np.select([M["mood"] > 0, M["mood"] < 0], ["😊 позитив", "😟 негатив"], "😐 нейтрально")
banner("Настроение сообщений", "🌡", "#F43F5E", "#F59E0B")
mood_counts = M["mood_label"].value_counts().reindex(["😊 позитив", "😐 нейтрально", "😟 негатив"]).fillna(0)
fig = plt.figure(figsize=(13, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.pie(mood_counts.values, labels=["позитив", "нейтрально", "негатив"], colors=["#34D399", "#CBD5E1", "#F87171"],
        autopct="%1.0f%%", startangle=90, wedgeprops=dict(width=.45))
ax1.set_title("Тон сообщений")
if HAS_DATE:
    ax2 = fig.add_subplot(1, 2, 2)
    mood_t = M.set_index("date")["mood"].resample("MS" if period_days and period_days > 90 else "W").mean()
    ax2.plot(mood_t.index, mood_t.values, marker="o", color="#F43F5E")
    ax2.axhline(0, color="#999", lw=1, ls="--")
    ax2.set_title("Средний «индекс настроения» во времени")
    plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=8)
plt.tight_layout()
show("Настроение")
print("⚠️ Оценка по словарю из ~60 корней: показывает общую тенденцию, а не точный смысл каждого сообщения.")


# ============================================================
# 10. 📊 Метрики (просмотры, реакции и т.п.)
# ============================================================
if ENG_COLS:
    banner("Метрики вовлечённости", "📊", "#0EA5E9", "#6366F1")
    for c in ENG_COLS:
        s = pd.to_numeric(M[c], errors="coerce")
        if s.notna().sum() < 3:
            continue
        print(f"{c}: всего {fmt(int(s.sum()))}, в среднем {s.mean():.1f}, максимум {fmt(int(s.max()))}")
        best = M.loc[s.nlargest(3).index]
        for _, r in best.iterrows():
            print(f"   ⭐ {int(r[c])} — {r['text'][:110].replace(chr(10), ' ')}")
    main = ENG_COLS[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(M["length"], pd.to_numeric(M[main], errors="coerce"), alpha=.35, s=14, color="#6366F1")
    ax.set_xscale("log"); ax.set_xlabel("длина сообщения"); ax.set_ylabel(main)
    ax.set_title(f"Связь длины сообщения и «{main}»")
    show("Метрики")


# ============================================================
# 11. 🃏 Карточки сообщений, поиск и тренды
# ============================================================
def card(row, query=None, color="#4F46E5"):
    meta = []
    if HAS_DATE and pd.notna(row["date"]):
        meta.append(f"{row['date']:%d.%m.%Y %H:%M}" if HAS_TIME else f"{row['date']:%d.%m.%Y}")
    if HAS_AUTHOR:
        meta.append(html_lib.escape(str(row["author"])))
    text = row["text"]
    body = html_lib.escape(text[:1200]) + ("…" if len(text) > 1200 else "")
    if query:
        body = re.sub(f"({re.escape(html_lib.escape(query))})", r"<mark>\1</mark>", body, flags=re.I)
    body = body.replace("\n", "<br>")
    return (f"<div style='border-left:4px solid {color};background:#F8FAFC;color:#111;padding:10px 14px;"
            f"margin:8px 0;border-radius:8px;font-family:sans-serif;font-size:14px'>"
            f"<div style='font-size:12px;color:#64748B;margin-bottom:4px'>{' · '.join(meta)}</div>{body}</div>")


def random_message(min_len=40):
    pool = M[M["length"] >= min_len]
    row = (pool if len(pool) else M).sample(1).iloc[0]
    display(HTML(card(row, color=random.choice(PALETTE))))


def search(query, author=None, limit=25, plot=True):
    """search('экзамен') — найти сообщения с текстом; author='Имя' — только этого автора."""
    mask = M["text"].str.contains(re.escape(query), case=False, na=False)
    if author and author != "Все":
        mask &= M["author"] == author
    res = M[mask]
    display(HTML(f"<b>Найдено: {fmt(len(res))}</b> по запросу «{html_lib.escape(query)}»"))
    if plot and HAS_DATE and len(res) > 1:
        rule = "MS" if period_days and period_days > 120 else "W"
        cnt = res.set_index("date").resample(rule).size()
        fig, ax = plt.subplots(figsize=(11, 2.6))
        ax.bar(cnt.index, cnt.values, width=20 if rule == "MS" else 5, color="#818CF8")
        ax.set_title(f"Когда упоминали «{query}»")
        plt.show()
    display(HTML("".join(card(r, query) for _, r in res.head(limit).iterrows())))
    if len(res) > limit:
        print(f"…показаны первые {limit}. Все результаты: export_excel('{query}')")
    return res


def word_trend(words, freq="M"):
    """word_trend('экзамен, сессия', freq='M') — частота слов по дням (D), неделям (W) или месяцам (M)."""
    if not HAS_DATE:
        print("В файле нет дат — тренд построить нельзя."); return
    if isinstance(words, str):
        words = [w.strip() for w in words.split(",") if w.strip()]
    rule = {"D": "D", "W": "W", "M": "MS"}.get(freq, "MS")
    base = M.set_index("date").resample(rule).size().index
    low = M["text"].str.lower().str.replace("ё", "е")
    fig, ax = plt.subplots(figsize=(12, 4))
    for w, color in zip(words, PALETTE):
        hit = M.loc[low.str.contains(re.escape(w.lower().replace("ё", "е")), na=False), "date"].dropna()
        cnt = pd.Series(1, index=hit.sort_values().values).resample(rule).sum() if len(hit) else pd.Series(dtype=float)
        cnt = cnt.reindex(base, fill_value=0)
        ax.plot(cnt.index, cnt.values, marker="o", ms=3, lw=2, color=color, label=f"{w} ({len(hit)})")
    ax.legend(); ax.set_title("Тренд упоминаний")
    plt.show()


def author_card(name):
    """author_card('Имя') — профиль автора."""
    sub = M[M["author"] == name]
    if sub.empty:
        print("Автор не найден"); return
    tw = Counter(w for ts in sub["tokens"] for w in ts).most_common(12)
    items = [("💬", fmt(len(sub)), "сообщений"),
             ("📏", fmt(int(sub["length"].mean())), "символов в среднем"),
             ("🏅", f"#{list(M['author'].value_counts().index).index(name) + 1}", "место по активности")]
    if HAS_DATE:
        items.append(("📅", f"{sub['date'].min():%d.%m.%Y}", "первое сообщение"))
        if HAS_TIME:
            items.append(("⏰", f"{int(sub['date'].dt.hour.mode().iloc[0]):02d}:00", "любимый час"))
    display(HTML(f"<h3 style='font-family:sans-serif'>👤 {html_lib.escape(name)}</h3>"))
    stat_cards(items)
    display(HTML("<p style='font-family:sans-serif'><b>Любимые слова:</b> " +
                 ", ".join(f"{html_lib.escape(w)} ({c})" for w, c in tw) + "</p>"))
    display(HTML(card(sub.sample(1).iloc[0])))


def export_excel(query=None, author=None, path="/content/messages_export.xlsx"):
    """export_excel('экзамен') — сохранить найденные сообщения в Excel и скачать."""
    res = M
    if query:
        res = res[res["text"].str.contains(re.escape(query), case=False, na=False)]
    if author:
        res = res[res["author"] == author]
    cols = ["date", "author", "text", "length", "mood_label"] + ENG_COLS
    out = res[cols].rename(columns={"date": "Дата", "author": "Автор", "text": "Текст",
                                    "length": "Длина", "mood_label": "Настроение"})
    out.to_excel(path, index=False)
    print(f"✅ Сохранено {fmt(len(out))} сообщений: {path}")
    if IN_COLAB:
        colab_files.download(path)


banner("Сообщение дня", "🎲", "#10B981", "#84CC16")
random_message()


# ============================================================
# 12. 🎛 Интерактивная панель
# ============================================================
banner("Интерактивная панель", "🎛", "#4F46E5", "#EC4899")
if HAS_WIDGETS:
    authors_list = ["Все"] + list(M["author"].value_counts().index[:300]) if HAS_AUTHOR else ["Все"]

    q_txt = widgets.Text(placeholder="слово или фраза", description="Найти:")
    q_auth = widgets.Dropdown(options=authors_list, description="Автор:")
    q_btn = widgets.Button(description="Искать", icon="search", button_style="primary")
    q_out = widgets.Output()

    def on_search(_):
        with q_out:
            q_out.clear_output()
            if q_txt.value.strip():
                search(q_txt.value.strip(), q_auth.value)
    q_btn.on_click(on_search)

    default_words = ", ".join(w for w, _ in WORDS.most_common(3))
    t_txt = widgets.Text(value=default_words, description="Слова:", layout=widgets.Layout(width="420px"))
    t_freq = widgets.ToggleButtons(options=[("Дни", "D"), ("Недели", "W"), ("Месяцы", "M")], value="M")
    t_btn = widgets.Button(description="Построить", icon="line-chart", button_style="info")
    t_out = widgets.Output()

    def on_trend(_):
        with t_out:
            t_out.clear_output()
            word_trend(t_txt.value, t_freq.value)
    t_btn.on_click(on_trend)

    a_dd = widgets.Dropdown(options=authors_list[1:] or ["—"], description="Автор:")
    a_out = widgets.Output()

    def on_author(change=None):
        with a_out:
            a_out.clear_output()
            if HAS_AUTHOR:
                author_card(a_dd.value)
            else:
                print("В файле не найдена колонка с авторами.")
    a_dd.observe(on_author, "value")

    r_btn = widgets.Button(description="Ещё сообщение!", icon="random", button_style="success")
    r_out = widgets.Output()

    def on_random(_):
        with r_out:
            r_out.clear_output()
            random_message()
    r_btn.on_click(on_random)

    tabs = widgets.Tab()
    tabs.children = [
        widgets.VBox([widgets.HBox([q_txt, q_auth, q_btn]), q_out]),
        widgets.VBox([widgets.HBox([t_txt, t_btn]), t_freq, t_out]),
        widgets.VBox([a_dd, a_out]),
        widgets.VBox([r_btn, r_out]),
    ]
    for i, t in enumerate(["🔎 Поиск", "📈 Тренд слов", "👤 Профиль автора", "🎲 Случайное"]):
        tabs.set_title(i, t)
    display(tabs)
    on_author(); on_random(None)
else:
    print("ipywidgets недоступен — используй функции search(), word_trend(), author_card().")


# ============================================================
# 13. 📄 HTML-отчёт
# ============================================================
def make_report(path="/content/messages_report.html"):
    """make_report() — красивый HTML-отчёт со всеми графиками; скачивается автоматически."""
    css = ("<style>body{font-family:sans-serif;max-width:1100px;margin:30px auto;color:#1E293B;padding:0 16px}"
           "h1{color:#4F46E5}h2{border-bottom:2px solid #E0E7FF;padding-bottom:4px;margin-top:36px}"
           "table{border-collapse:collapse}td,th{border:1px solid #E2E8F0;padding:6px 10px}"
           "th{background:#4F46E5;color:white}img{max-width:100%;border-radius:8px}</style>")
    parts = [f"<h1>💬 Отчёт по сообщениям</h1><p>{html_lib.escape(os.path.basename(FILE_PATH))} · "
             f"сформирован {datetime.datetime.now():%d.%m.%Y %H:%M}</p>",
             "<h2>🏆 Главное</h2><table>" + "".join(
                 f"<tr><td><b>{html_lib.escape(k)}</b></td><td>{html_lib.escape(v)}</td></tr>"
                 for k, v in [("Сообщений", fmt(len(M)))] + records) + "</table>"]
    if TOPICS:
        parts.append("<h2>🧩 Темы</h2>" + pd.DataFrame(
            TOPICS, columns=["Тема", "Доля, %", "Ключевые слова"]).to_html(index=False))
    for title, img in FIGS:
        parts.append(f"<h2>{html_lib.escape(title)}</h2><img src='data:image/png;base64,{img}'>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>Отчёт</title>{css}</head>"
                f"<body>{''.join(parts)}</body></html>")
    print(f"✅ Отчёт сохранён: {path}")
    if IN_COLAB:
        colab_files.download(path)


banner("Готово! Команды для следующих ячеек", "🏁", "#1E3A8A", "#4F46E5")
display(Markdown("""
- `search("экзамен")` — найти сообщения (можно добавить `author="Имя"`)
- `word_trend("экзамен, сессия", freq="M")` — тренд слов по дням `D`, неделям `W` или месяцам `M`
- `author_card("Имя")` — профиль автора
- `random_message()` — случайное сообщение
- `export_excel("экзамен")` — выгрузить найденное в Excel
- `make_report()` — скачать HTML-отчёт со всеми графиками
- `M` — таблица сообщений (pandas DataFrame), `RAW` — исходный файл
"""))
