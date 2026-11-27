"""Benign text generators and corruption machinery for the corpus.

Standard library only. Every generator takes a `random.Random` and draws all
randomness from it, so output is deterministic for a seed.

Contents:
  * shared pools used by benign AND attack carriers (parameter names, cookie
    names, JSON keys, neutral host generator, identifiers)
  * generated English text from an inline vocabulary
  * generated text in many scripts (composed, not drawn from fixed sentences)
  * technical text: code one-liners, config lines, regexes, version
    constraints, header values, MIME types, paths, query strings, form
    bodies, cookies, CSV/TSV, log and stack-trace lines, dates and times
  * values with ordinary percent-escapes, JSON \\uXXXX escapes, HTML entities
  * `corrupt()` which applies exactly one structural damage
  * `benign_ok()` final check applied to every benign sample
"""

from __future__ import annotations

import base64
import json
import functools
import random
import re
from typing import Callable, List, Optional, Tuple


def _w(s: str) -> List[str]:
    return list(dict.fromkeys(s.split()))


# ==========================================================================
# English vocabulary (inline)
# ==========================================================================

NOUNS = _w("""
time year people way day man thing woman life child world school state family
student group country problem hand part place case week company system program
question work government number night point home water room mother area money
story fact month lot right study book eye job word business issue side kind head
house service friend father power hour game line end member law car city
community name president team minute idea kid body information back parent face
level office door health person art war history party result change morning
reason research girl guy moment air teacher force education foot boy age policy
process music market sense nation plan college interest death experience effect
class control care field development role effort rate heart show leader light
voice wife police mind price report decision son view relationship town road arm
difference value building action model season society tax director position
player record paper space ground form event official matter center couple site
project activity star table need court oil situation cost industry figure street
image phone data picture practice piece land product doctor wall patient worker
news test movie north love support technology step baby computer type attention
film tree source organization hair window evidence population training island
chair bed garden kitchen bottle glass cup plate spoon fork knife bread butter
cheese milk coffee tea juice sugar salt pepper rice pasta soup salad sandwich
pizza burger chicken beef pork fish egg apple banana orange grape lemon
strawberry cherry peach pear melon tomato potato onion garlic carrot bean corn
mushroom cake cookie chocolate candy dessert breakfast lunch dinner meal recipe
restaurant menu waiter bill tip receipt delivery package box envelope letter
stamp card gift present ticket seat flight airport plane train station bus taxi
bike bicycle boat ship truck engine wheel tire fuel parking garage bridge tunnel
highway traffic map direction corner block neighborhood village region province
capital border coast beach ocean sea lake river stream mountain hill valley
forest desert farm animal dog cat horse cow sheep pig bird duck goose rabbit
mouse bear wolf fox deer lion tiger elephant monkey snake frog turtle whale
dolphin shark insect bee butterfly spider flower rose grass leaf branch root seed
plant weather rain snow wind storm cloud sun moon sky temperature heat cold
spring summer autumn fall winter holiday vacation weekend birthday anniversary
wedding festival concert theater museum gallery library park zoo stadium gym
pool shop store mall supermarket bakery pharmacy hospital clinic bank hotel
apartment floor roof ceiling stairs elevator hallway bathroom bedroom closet
shelf drawer desk lamp sofa couch carpet curtain mirror clock frame pillow
blanket towel soap shampoo toothbrush toothpaste brush comb shirt pants jeans
dress skirt jacket coat sweater hat cap scarf glove sock shoe boot belt tie
button pocket wallet purse bag backpack suitcase umbrella watch ring necklace
bracelet earring glasses keyboard screen monitor laptop tablet printer scanner
camera speaker headphone microphone battery charger cable cord plug switch file
folder document spreadsheet slide presentation email message chat call meeting
conference agenda schedule calendar deadline task goal target budget invoice
payment account balance credit debit loan profit loss revenue sale discount
coupon offer deal fee charge refund exchange warranty shipping customer client
vendor supplier partner manager employee staff colleague boss intern candidate
interview resume salary bonus promotion career skill talent knowledge lesson
course exam grade degree diploma certificate university campus classroom
homework assignment essay chapter page paragraph sentence symbol sign signal
note comment reply review rating feedback survey poll vote election campaign
citizen freedom justice peace safety security risk danger threat accident
emergency injury pain disease illness fever cough medicine pill nurse dentist
surgeon therapy treatment appointment insurance contract agreement license permit
rule regulation standard guideline procedure method approach solution answer
option choice chance opportunity advantage benefit feature function tool device
machine equipment material metal wood plastic stone brick concrete sand dust dirt
smoke fire flame energy electricity gas coal network internet website link
browser app application software hardware server database cloud storage backup
update version release bug error ticket request response status code key
password username login profile setting preference theme color font size width
height length weight volume speed distance shape circle square triangle edge
surface middle top bottom front left inside outside beginning start finish
album artist author band bottle brand captain career cart category ceremony
champion channel character chart cheek chef chest choir climate coach collection
column comedy committee competition concept condition context contest corridor
costume cotton county courage cousin crew crowd culture currency curve cycle
dancer daughter decade defense degree demand deposit designer detail diet dinner
disaster discovery dish distance district draft drama drawing driver eagle earth
economy editor engineer episode estate exhibit expert fabric factory fan fashion
fence fiction finger flag fleet flavor fridge fruit funeral furniture galaxy
generation genre ghost giant glove grain grandmother guest guitar habit harbor
harvest headline helmet hero highlight hobby honey horizon host household
husband icon identity incident income index ingredient inn input insight
instance instrument inventory iron item jacket jar journal journey judge jungle
kettle kingdom ladder lane laundry lawyer layer league legend leisure lens lid
lifestyle limit liquid list lobby location lock logo lounge luggage magazine
mail mall manual margin marriage mask match meadow measure memory mention
merchant metro midnight milestone mineral minister mission mixture moment mood
motor mountain movement mug nature neighbor nephew niece noise novel nut object
occasion ocean opinion orchestra outcome outfit owner pack pad palace panel
parade passenger passport path pattern pavement peak pen pencil period pet
phrase pianist piano picnic pilot pipe planet platform player pleasure pocket
poem poet portrait post pot powder prayer premium pride prince princess print
priority prize producer professor progress property proposal prospect puzzle
quality quarter queen queue quote radio railway range ratio reader recipe
reference relief rent replacement reputation reserve resident resort resource
rhythm ribbon rider rival robot rock rocket routine row sailor sample satellite
sauce scale scene scholar scope score sculpture section sector segment seller
seminar series session shade shadow shelter shift shore shower sight silver
singer sister sketch skin slope snack soldier song soul spirit sponsor sport
spot stage stamp statement statue steak steel stick stock stomach strategy straw
strength string structure studio style subject suburb success suit summit
supply surgery surprise symbol talent tank taste teen temple tent term terrace
text texture theme theory thread throat thumb tide timber title toast toe
tone tongue tooth topic tour tourist tower toy track trade tradition trail
transport trap tray treasure trend trial tribe trip trophy tube tutor twin
uncle uniform unit universe vacuum valley van vehicle venue verse vessel victim
video view village violin vision visitor volunteer wage wagon walk warehouse
wave wealth web week wheat whistle widow wing wire wisdom witness wonder workshop
writer yard youth zone
""")

VERBS = _w("""
be have do say go get make know think take see come want look use find give tell
work call try ask need feel become leave put mean keep let begin seem help talk
turn start show hear play run move like live believe hold bring happen write
provide sit stand lose pay meet include continue set learn change lead understand
watch follow stop create speak read allow add spend grow open walk win offer
remember love consider appear buy wait serve send expect build stay fall cut
reach remain suggest raise pass sell require report decide pull return explain
hope develop carry break receive agree support hit produce eat cover catch draw
choose cause point listen realize place close involve increase finish arrive
mention describe improve prepare enjoy share visit check order book cancel
confirm reply forward attach download upload install update restart refresh save
delete edit copy paste print scan sign submit apply join register subscribe
unsubscribe browse search filter sort select drop insert compare measure count
calculate estimate plan schedule organize arrange manage handle fix repair clean
wash cook bake boil fry mix pour chop slice taste smell touch hug kiss smile
laugh cry shout whisper sing dance swim climb jump ride drive fly travel explore
discover invent design paint sketch photograph film record publish post comment
mute text wake sleep rest relax exercise train practice study teach review test
grade fail graduate hire retire quit resign promote earn invest borrow lend owe
donate deliver ship pack unpack wrap lock unlock enter exit push lift throw
kick miss beat score celebrate thank apologize forgive complain argue discuss
debate vote elect protest announce declare promise warn advise recommend invite
welcome greet introduce accept reject approve deny refuse admit prove solve
answer question wonder guess imagine dream wish worry fear hate prefer notice
recognize identify forget recall remind repeat pause resume hurry rush delay
postpone attend host sponsor launch release migrate deploy configure monitor
debug compile merge commit refactor document translate summarize outline
highlight underline bookmark tag rename archive restore recover sync connect
disconnect pair charge unplug rent lease adopt adjust admire afford aim announce
appreciate assemble assign assist assume attract avoid bake balance bathe beg
behave bend bless blink boast bounce bow breathe brew browse bury calm care chase
cheer chew clap coach collect combine comfort commute compete complete concentrate
contribute convince cooperate correct cough crawl cross crush cure dare decorate
defend deliver depend deserve destroy detect divide double doubt drag drain dress
drift drill earn educate embrace emerge employ encourage engage enhance ensure
entertain escape evaluate examine excuse expand explore express extend fade fasten
feed fetch fill fit flash float flow fold forecast frame freeze gather gaze glow
govern grab greet grind guide hang harvest heal heat hide hike hum hunt ignore
illustrate impress inform inspect inspire instruct interrupt iron juggle knit
knock label land last lean leap lie light limit link load locate march mark marry
melt mend mine motivate multiply nod obey observe obtain occupy operate overcome
own paddle park participate peel perform permit persuade pick pinch plant polish
pop practise praise pray predict prefer present pretend prevent print proceed
progress protect provide punch qualify race rain rank rate recycle reduce reflect
regret relate rely remove rescue research reserve respond retain reveal reward
ring rinse roast roll rub sail satisfy scatter scrub seal secure seek settle
shake shape shine shop sigh sink sip skate ski skip slide slip smash sneeze sniff
snow soak sort spare spark spell spill spin splash spray spread squeeze stack stamp
stare steer stir stitch stretch stroll struggle succeed suffer supply surround
survive suspect swing tackle tap tease thrive tickle tidy tour trace trade trust
tug tumble twist type unite unload upgrade urge value vanish verify wander warm
water wave weigh whistle wipe wobble wrestle yawn yell zoom
""")

VERBS_PAST = _w("""
was were had did said went got made knew thought took saw came wanted looked used
found gave told worked called tried asked needed felt became left meant kept
began seemed helped talked turned started showed heard played ran moved liked
lived believed brought happened wrote sat stood lost paid met included continued
learned changed led understood watched followed stopped created spoke allowed
added spent grew opened walked won offered remembered bought waited sent expected
built stayed fell reached decided returned explained received agreed chose
arrived ordered booked cancelled confirmed shipped delivered fixed cleaned cooked
baked visited enjoyed shared checked updated saved deleted signed joined posted
replied planned visited painted traveled packed carried dropped picked tasted
""")

VERBS_ING = _w("""
having doing going getting making working looking using trying thinking coming
waiting running reading writing playing watching listening cooking traveling
shopping planning moving building testing hiking baking learning teaching driving
singing dancing painting swimming walking talking sleeping eating drinking
""")

ADJS = _w("""
good new first last long great little own other old right big high different
small large next early young important few public bad same able free sure clear
full special easy strong certain real best better hard whole possible major
personal current national natural physical local late simple short single main
recent final happy serious ready nice fresh quick slow fast bright dark light
heavy soft warm cold hot cool dry wet clean dirty empty busy quiet loud safe
dangerous cheap expensive rich poor beautiful pretty ugly cute handsome tall tiny
huge wide narrow deep shallow thick thin round square flat smooth rough sharp
sweet sour bitter salty spicy delicious tasty healthy sick tired hungry thirsty
angry sad glad proud lucky brave calm careful kind friendly polite rude funny
boring interesting exciting amazing awesome wonderful terrible awful horrible
perfect excellent fantastic brilliant fine okay average normal regular typical
usual strange weird odd rare common popular famous unknown modern ancient classic
traditional digital online offline mobile wireless global international domestic
foreign official private secret open closed available unavailable missing broken
damaged used original extra additional optional required necessary basic advanced
complex detailed brief complete partial total exact accurate correct wrong false
true valid invalid active inactive pending approved rejected urgent upcoming
previous following daily weekly monthly annual yearly hourly instant automatic
manual direct indirect positive negative neutral central northern southern
eastern western upper lower inner outer rear minor primary secondary green red
blue yellow orange purple pink brown black white gray silver golden wooden
plastic metal cotton leather organic frozen raw cooked baked fried grilled
roasted vegan vegetarian lightweight waterproof portable adjustable reusable
durable comfortable stylish elegant casual formal ample anxious bold bumpy chilly
clever cozy crisp crowded curious dusty eager elderly energetic fancy fierce
fluffy foggy fragile gentle gigantic glossy graceful greasy grumpy hollow honest
humble icy jolly juicy keen lazy lively lonely loyal magnetic massive mature
messy mild misty muddy mysterious nervous noisy obvious patient peaceful plain
pleasant precious punctual rapid rare relaxed reliable rusty scenic shiny shy
silky silly sleepy slim sloppy snowy solid sparkling spotless steady sticky
stormy sturdy sunny superb tender tidy tough tricky vast vivid wise witty worried
young zealous
""")

ADVS = _w("""
very really just also only still even now then here there always never often
sometimes usually rarely quickly slowly easily finally actually probably maybe
perhaps almost already soon later today tomorrow yesterday tonight again together
away back up down out off over around nearly quite rather too enough exactly
especially certainly definitely absolutely completely totally fully partly mostly
mainly simply clearly obviously honestly seriously literally basically generally
normally recently currently previously immediately suddenly eventually gradually
carefully happily sadly luckily unfortunately hopefully anyway otherwise instead
however therefore meanwhile afterwards everywhere somewhere anywhere nowhere
forever once twice early late far well badly upstairs downstairs abroad ahead
apart barely briefly calmly closely deeply directly equally fairly firmly freely
gently greatly highly kindly largely lately loosely loudly merely neatly newly
openly politely properly quietly rapidly regularly safely sharply smoothly softly
strongly truly warmly widely
""")

FUNC = _w("""
the a an this that these those my your his her its our their some any every each
no all both either neither much many more most less least few several another
such what which who whom whose when where why how whether if unless because since
although though while as than so but and or nor yet for of in on at by to from
with without about above below under between among through during before after
until against across along around behind beside beyond into near onto past
toward towards upon via within I you he she it we they me him us them myself
yourself one nothing something everything anything someone everyone anyone nobody
""")

VOCAB: List[str] = list(dict.fromkeys(NOUNS + VERBS + VERBS_PAST + VERBS_ING + ADJS + ADVS + FUNC))
IDENT_WORDS: List[str] = [w for w in NOUNS if w.isalpha() and len(w) <= 10]

FIRST_NAMES = _w("""
James Mary John Patricia Robert Jennifer Michael Linda David Elizabeth William
Barbara Richard Susan Joseph Jessica Thomas Sarah Charles Karen Christopher Lisa
Daniel Nancy Matthew Betty Anthony Sandra Mark Margaret Donald Ashley Steven
Kimberly Andrew Emily Paul Donna Joshua Michelle Kenneth Carol Kevin Amanda Brian
Melissa George Deborah Timothy Stephanie Ronald Rebecca Jason Sharon Edward Laura
Jeffrey Cynthia Ryan Dorothy Jacob Amy Gary Kathleen Nicholas Angela Eric Emma
Stephen Brenda Justin Nicole Scott Anna Brandon Samantha Benjamin Katherine
Samuel Rachel Patrick Frank Janet Maria Jack Olivia Heather Tyler Catherine Aaron
Julie Adam Victoria Nathan Henry Lauren Peter Ruth Kyle Noah Ethan Hannah
Christian Andrea Megan Austin Madison Sean Grace Sophia Isabella Logan Julia
Charlotte Natalie Diana Sofia Wei Yuki Mohammed Olga Hiroshi Fatima Chen Priya
Lars Amara Aisha Mateo Lucia Diego Camila Santiago Valentina Ahmed Omar Layla
Yusuf Zainab Ravi Ananya Arjun Hana Minji Jiho Haruto Sakura Ren Mei Jun Xin
Ivan Dmitri Natasha Sven Ingrid Freya Pierre Luca Giulia Marco Chiara Hans Greta
Piotr Kwame Chidi Ngozi Thabo Siobhan Niamh Aoife Rhys Kai Leilani Tariq Noor
Mehmet Elif Emre
""")

LAST_NAMES = _w("""
Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez
Hernandez Lopez Gonzalez Wilson Anderson Thomas Taylor Moore Jackson Martin Lee
Perez Thompson White Harris Sanchez Clark Ramirez Lewis Robinson Walker Young
Allen King Wright Scott Torres Nguyen Hill Flores Green Adams Nelson Baker Hall
Rivera Campbell Mitchell Carter Roberts O'Brien O'Connor O'Neill O'Malley
O'Reilly d'Angelo D'Souza McDonald MacLeod Fitzgerald Kim Park Choi Nakamura
Tanaka Suzuki Sato Ivanov Petrov Smirnova Okafor Mensah Al-Farsi Haddad Khan
Patel Sharma Singh Zhao Wang Li Liu Chen Kowalski Novak Jensen Larsen Rossi
Bianchi Schmidt Fischer Weber Dubois Moreau Silva Santos Costa
""")
FIRST_NAMES_ACCENTED = _w("José Zoë Søren Björn Chloé Renée François Ángel Inés Jürgen Łukasz Mónica Thérèse Ömer Dvořák Ñuño")
LAST_NAMES_ACCENTED = _w("Müller Gonçalves Núñez Peña Lefèvre Øster Kovač Šimić Ångström Çelik Yılmaz Dvořák Söderberg")


def name(rng: random.Random) -> str:
    if rng.random() < 0.12:
        f = rng.choice(FIRST_NAMES_ACCENTED)
    else:
        f = rng.choice(FIRST_NAMES)
    if rng.random() < 0.1:
        l = rng.choice(LAST_NAMES_ACCENTED)
    else:
        l = rng.choice(LAST_NAMES)
    r = rng.random()
    if r < 0.6:
        return f + " " + l
    if r < 0.75:
        return f
    if r < 0.85:
        return "%s %s. %s" % (f, rng.choice("ABCDEFGHJKLMNPRSTW"), l)
    return l + ", " + f


# ==========================================================================
# Shared pools (benign AND attack carriers)
# ==========================================================================

PARAM_NAMES = _w("""
q id page search name user lang sort order category ref redirect url next file
path view token sku pid uid action type format callback limit offset tab filter
country lat lng query s p v k t n x y z a b c d e f i j m r w
utm_source utm_medium utm_campaign utm_term utm_content gclid fbclid ref_src
session_id sid lang_code locale currency region city zip postal state address
email phone first_name last_name username login password confirm remember
title subject message comment body text content description note summary tags
tag color size qty quantity price min_price max_price from to start end date
time year month day week since until before after cursor page_size per_page
sort_by order_by dir asc desc view_mode layout theme debug mode preview draft
status state_id item item_id product product_id cart cart_id checkout coupon
promo code ticket order_id invoice account account_id org org_id team team_id
project project_id repo branch commit tab_id section anchor slug lang_id
return_to returnUrl continue dest destination target goto source src origin
width height w h dpr quality fmt ext download inline attachment filename
version v1 api_key key client_id scope state nonce response_type grant_type
""")

COOKIE_NAMES = _w("""
sid session sessionid SESSID PHPSESSID JSESSIONID ASP.NET_SessionId connect.sid
auth token access_token refresh_token csrftoken XSRF-TOKEN _csrf uid user_id
cart cart_id pref prefs lang locale theme tz consent cookie_consent _ga _gid
_gat _fbp ab_test experiment visitor_id device_id remember_me last_visit
""")

JSON_KEYS = list(dict.fromkeys(PARAM_NAMES + _w("""
comment message value input data result items meta error code detail details
username password email phone address city country title description body text
content note notes label caption bio display_name nickname avatar url link
created_at updated_at timestamp count total score rating enabled active
""")))

TLDS = _w("""
com com com com net org io co dev app ai info biz me tv us uk de fr jp kr cn ru
br in au ca es it nl se ch pl co.uk com.au co.jp com.br edu gov cloud shop
online site tech xyz store blog news
""")
SUBDOMAINS = _w("""
www www www api cdn static img images m mobile shop blog mail app auth login
docs dev staging beta help support status assets media files news portal account
""")

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_ALNUM = _LOWER + _LOWER.upper() + "0123456789"
_HEX = "0123456789abcdef"
_SYL_C = "b c d f g h j k l m n p r s t v w z br ch cl cr dr fl fr gl gr pl pr sh st str th tr".split()
_SYL_V = "a e i o u a e i o ai ea io ou".split()


def rand_hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_HEX) for _ in range(n))


def rand_alnum(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_ALNUM) for _ in range(n))


def rand_b64(rng: random.Random, n: int, url: bool = False) -> str:
    alpha = _ALNUM + ("-_" if url else "+/")
    return "".join(rng.choice(alpha) for _ in range(n))


def brand(rng: random.Random) -> str:
    """Pronounceable invented word (company / product / domain-like)."""
    s = "".join(rng.choice(_SYL_C) + rng.choice(_SYL_V) for _ in range(rng.randint(1, 3)))
    if rng.random() < 0.3:
        s += rng.choice(("ly", "fy", "io", "er", "on", "ex", "ia", "hub", "box", "lab"))
    return s


def ident(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.35:
        return rng.choice(IDENT_WORDS)
    if r < 0.5:
        return rng.choice(IDENT_WORDS) + "_" + rng.choice(IDENT_WORDS)
    if r < 0.62:
        a, b = rng.choice(IDENT_WORDS), rng.choice(IDENT_WORDS)
        return a + b.capitalize()
    if r < 0.74:
        return rng.choice(IDENT_WORDS) + str(rng.randint(1, 999))
    if r < 0.86:
        return brand(rng)
    return "".join(rng.choice(_LOWER) for _ in range(rng.randint(2, 8)))


def ipv4(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.3:
        return "10.%d.%d.%d" % (rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254))
    if r < 0.5:
        return "192.168.%d.%d" % (rng.randint(0, 255), rng.randint(1, 254))
    if r < 0.6:
        return "172.%d.%d.%d" % (rng.randint(16, 31), rng.randint(0, 255), rng.randint(1, 254))
    return "%d.%d.%d.%d" % (rng.randint(1, 223), rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254))


def ipv6(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return "2001:db8:%s::%s" % (rand_hex(rng, 4).lstrip("0") or "0", rand_hex(rng, 3))
    if r < 0.7:
        return "fe80::%s:%s:%s:%s" % tuple(rand_hex(rng, 4) for _ in range(4))
    return ":".join(rand_hex(rng, 4) for _ in range(8))


def neutral_host(rng: random.Random) -> str:
    """Neutral host name shared by benign and attack samples."""
    r = rng.random()
    if r < 0.06:
        return ipv4(rng)
    if r < 0.09:
        return "localhost"
    base = rng.choice((brand(rng), rng.choice(IDENT_WORDS), brand(rng),
                       rng.choice(IDENT_WORDS) + rng.choice(IDENT_WORDS),
                       rng.choice(IDENT_WORDS) + "-" + rng.choice(IDENT_WORDS)))
    host = base + "." + rng.choice(TLDS)
    if rng.random() < 0.55:
        host = rng.choice(SUBDOMAINS) + "." + host
    return host


def url_path(rng: random.Random, depth: Optional[int] = None) -> str:
    n = rng.randint(0, 4) if depth is None else depth
    segs = []
    for _ in range(n):
        r = rng.random()
        if r < 0.4:
            segs.append(rng.choice(IDENT_WORDS))
        elif r < 0.55:
            segs.append("-".join(rng.choice(VOCAB).lower() for _ in range(rng.randint(2, 5))
                                 ).replace("'", ""))
        elif r < 0.7:
            segs.append(str(rng.randint(1, 999999)))
        elif r < 0.8:
            segs.append(rng.choice(("api", "v1", "v2", "v3", "en", "en-us", "de", "static",
                                    "assets", "u", "p", "s", "admin", "account", "app")))
        elif r < 0.9:
            segs.append(rand_alnum(rng, rng.randint(6, 12)))
        else:
            segs.append(filename(rng))
    p = "/" + "/".join(segs)
    if segs and rng.random() < 0.15:
        p += "/"
    return p


FILE_EXTS = _w("""
txt md pdf doc docx xls xlsx ppt pptx csv tsv json xml yaml yml toml ini cfg conf
log png jpg jpeg gif webp svg ico mp3 mp4 mov avi wav zip tar gz tgz 7z rar html
htm css js mjs ts tsx jsx py rb go rs java kt c h cpp hpp cs swift php sql sh
odt rtf epub psd ai sketch fig bak tmp lock
""")


def filename(rng: random.Random) -> str:
    stem = rng.choice((
        rng.choice(IDENT_WORDS),
        rng.choice(IDENT_WORDS) + "_" + rng.choice(IDENT_WORDS),
        rng.choice(IDENT_WORDS) + "-" + str(rng.randint(2000, 2026)),
        "IMG_%04d" % rng.randint(1, 9999),
        "%s %s" % (rng.choice(IDENT_WORDS).capitalize(), rng.choice(IDENT_WORDS)),
        "report_q%d" % rng.randint(1, 4),
        rand_alnum(rng, 8),
    ))
    return stem + "." + rng.choice(FILE_EXTS)


def url(rng: random.Random) -> str:
    u = rng.choice(("https://", "https://", "https://", "http://")) + neutral_host(rng)
    if rng.random() < 0.08:
        u += ":%d" % rng.choice((8080, 8443, 3000, 5000, 8000, 9000))
    u += url_path(rng)
    if rng.random() < 0.4:
        u += "?" + query_string(rng, rng.randint(1, 4))
    if rng.random() < 0.08:
        u += "#" + rng.choice(IDENT_WORDS)
    return u


def email(rng: random.Random) -> str:
    f = rng.choice(FIRST_NAMES).lower()
    l = rng.choice(LAST_NAMES).lower().replace("'", "").replace(" ", "")
    local = rng.choice((f, f + "." + l, f[0] + l, f + str(rng.randint(1, 99)), f + "_" + l,
                        f + "+" + rng.choice(IDENT_WORDS)))
    dom = rng.choice(("gmail.com", "outlook.com", "yahoo.com", "proton.me", "icloud.com"))
    if rng.random() < 0.5:
        dom = brand(rng) + "." + rng.choice(TLDS)
    return local + "@" + dom


# ==========================================================================
# English text with realistic punctuation
# ==========================================================================

_DETS = ("the", "the", "a", "this", "that", "my", "our", "your", "their", "his", "her", "every", "one")
_DETS_PL = ("the", "some", "these", "those", "our", "your", "many", "several", "two", "three", "all")
_PRONOUNS = ("I", "you", "we", "they", "he", "she", "it", "someone", "everyone")
_MODALS = ("will", "can", "should", "could", "would", "might", "must")
_CONTRACTIONS = ("I'm", "don't", "it's", "we'll", "they've", "can't", "won't", "isn't",
                 "you're", "that's", "there's", "let's", "didn't", "I've", "she's", "wasn't",
                 "we're", "couldn't", "shouldn't", "I'd", "what's", "aren't", "haven't")
_PREPS = ("in", "on", "at", "by", "for", "with", "from", "to", "about", "after", "before",
          "near", "under", "over", "during", "without", "between", "around", "into")
_INTERJ = ("Wow", "Honestly", "Well", "Okay", "Oh", "Hey", "Thanks", "Sadly", "Luckily",
           "By the way", "Anyway", "Also", "Btw", "FYI", "PS")


def _plural(n: str) -> str:
    if n.endswith(("s", "sh", "ch", "x")):
        return n + "es"
    if n.endswith("y") and len(n) > 1 and n[-2] not in "aeiou":
        return n[:-1] + "ies"
    return n + "s"


def _np(rng: random.Random) -> str:
    noun = rng.choice(NOUNS)
    adj = rng.choice(ADJS) + " " if rng.random() < 0.4 else ""
    if rng.random() < 0.3:
        return rng.choice(_DETS_PL) + " " + adj + _plural(noun)
    det = rng.choice(_DETS)
    if det == "a" and (adj or noun)[:1] in "aeiou":
        det = "an"
    if rng.random() < 0.08:
        return name(rng).split(",")[0] + "'s " + adj + noun
    return det + " " + adj + noun


def _subject(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.35:
        return rng.choice(_PRONOUNS)
    if r < 0.5:
        return name(rng).split(",")[0]
    return _np(rng)


def _clause(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.35:
        s = "%s %s %s" % (_subject(rng), rng.choice(VERBS_PAST), _np(rng))
    elif r < 0.5:
        s = "%s %s %s %s" % (_subject(rng), rng.choice(_MODALS), rng.choice(VERBS), _np(rng))
    elif r < 0.6:
        s = "%s %s %s" % (rng.choice(_CONTRACTIONS), rng.choice(VERBS), _np(rng))
    elif r < 0.7:
        s = "%s %s is %s" % (rng.choice(("the", "this", "that", "our", "my")), rng.choice(NOUNS),
                             rng.choice(ADJS))
    elif r < 0.8:
        s = "%s %s %s" % (rng.choice(VERBS).capitalize() if rng.random() < 0.5 else "please " +
                          rng.choice(VERBS), _np(rng), rng.choice(_PREPS) + " " + _np(rng))
    elif r < 0.9:
        s = "%s %s %s %s" % (_subject(rng), rng.choice(ADVS), rng.choice(VERBS_PAST), _np(rng))
    else:
        s = "there %s %s %s" % (rng.choice(("is", "was", "are", "were")), _np(rng),
                                rng.choice(_PREPS) + " " + _np(rng))
    if rng.random() < 0.35:
        s += " " + rng.choice(_PREPS) + " " + _np(rng)
    return s


def money(rng: random.Random) -> str:
    amt = rng.choice(("%d" % rng.randint(1, 999), "%d.%02d" % (rng.randint(0, 999), rng.randint(0, 99)),
                      "{:,}".format(rng.randint(1000, 250000))))
    sym = rng.choice(("$", "$", "€", "£", "¥", "₹", "CHF ", "US$", "A$", "₩"))
    if rng.random() < 0.2:
        return amt + " " + rng.choice(("USD", "EUR", "GBP", "JPY", "KRW"))
    return sym + amt


def percent(rng: random.Random) -> str:
    return rng.choice(("%d%%" % rng.randint(1, 100), "%.1f%%" % (rng.random() * 100),
                       "%d %%" % rng.randint(1, 99), "-%d%%" % rng.randint(1, 60),
                       "+%d%%" % rng.randint(1, 300)))


def clock(rng: random.Random) -> str:
    h, m = rng.randint(0, 23), rng.randint(0, 59)
    return rng.choice(("%d:%02d" % (h, m), "%02d:%02d" % (h, m),
                       "%d:%02d %s" % (h % 12 or 12, m, rng.choice(("am", "pm", "AM", "PM"))),
                       "%dpm" % rng.randint(1, 11), "%d:%02d:%02d" % (h, m, rng.randint(0, 59))))


_MONTHS = ("January February March April May June July August September October November December").split()
_DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()


def date_time(rng: random.Random) -> str:
    y, mo, d = rng.randint(1990, 2027), rng.randint(1, 12), rng.randint(1, 28)
    h, mi, s = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    return rng.choice((
        "%04d-%02d-%02d" % (y, mo, d),
        "%04d-%02d-%02dT%02d:%02d:%02dZ" % (y, mo, d, h, mi, s),
        "%04d-%02d-%02dT%02d:%02d:%02d.%03d+%02d:00" % (y, mo, d, h, mi, s, rng.randint(0, 999), rng.randint(0, 12)),
        "%02d/%02d/%04d" % (mo, d, y),
        "%02d.%02d.%04d" % (d, mo, y),
        "%d %s %04d" % (d, _MONTHS[mo - 1], y),
        "%s %d, %04d" % (_MONTHS[mo - 1], d, y),
        "%s, %02d %s %04d %02d:%02d:%02d GMT" % (_DAYS[rng.randint(0, 6)][:3], d, _MONTHS[mo - 1][:3], y, h, mi, s),
        "%04d/%02d/%02d %02d:%02d" % (y, mo, d, h, mi),
        "%d" % rng.randint(946684800, 1893456000),
        "%d" % rng.randint(946684800000, 1893456000000),
        "%s %d" % (_MONTHS[mo - 1][:3], d),
        "%04d年%d月%d日" % (y, mo, d),
        "Q%d %04d" % (rng.randint(1, 4), y),
    ))


def _decor(rng: random.Random) -> str:
    """Inline extras that make English text realistic."""
    r = rng.random()
    if r < 0.12:
        return percent(rng)
    if r < 0.24:
        return money(rng)
    if r < 0.32:
        return clock(rng)
    if r < 0.38:
        return rng.choice(("%d/5" % rng.randint(1, 5), "%.1f/5" % (rng.randint(10, 50) / 10),
                           "%d/10" % rng.randint(1, 10), "%d out of %d" % (rng.randint(1, 9), 10),
                           "%d stars" % rng.randint(1, 5)))
    if r < 0.45:
        return "@" + rng.choice((ident(rng), rng.choice(FIRST_NAMES).lower() + str(rng.randint(1, 99))))
    if r < 0.52:
        return "#" + rng.choice((rng.choice(IDENT_WORDS), rng.choice(IDENT_WORDS) + rng.choice(IDENT_WORDS).capitalize(),
                                 rng.choice(IDENT_WORDS) + str(rng.randint(2010, 2026))))
    if r < 0.6:
        a, b = rng.choice(LAST_NAMES), rng.choice(LAST_NAMES)
        return rng.choice(("%s & %s" % (a, b), "%s & Sons" % a, "%s & Co." % a,
                           "%s & %s" % (rng.choice(FIRST_NAMES), rng.choice(FIRST_NAMES))))
    if r < 0.66:
        return date_time(rng)
    if r < 0.72:
        return rng.choice(("%d" % rng.randint(0, 5000), "%d-%d" % (rng.randint(1, 20), rng.randint(21, 99)),
                           "#%d" % rng.randint(1, 99999), "No. %d" % rng.randint(1, 500),
                           "%dx" % rng.randint(2, 10), "%dkm" % rng.randint(1, 900),
                           "%d°C" % rng.randint(-20, 40), "%d kg" % rng.randint(1, 120)))
    if r < 0.76:
        return url(rng) if rng.random() < 0.6 else email(rng)
    if r < 0.8:
        return emoji(rng)
    return rng.choice(VOCAB)


def english_sentence(rng: random.Random) -> str:
    parts = [_clause(rng)]
    for _ in range(rng.choices((0, 1, 2, 3), (30, 40, 20, 10))[0]):
        sep = rng.choice((", ", ", and ", ", but ", "; ", "; ", " -- ", " - ", " — ", ": ",
                          "... ", " (", ", so ", " because ", " while ", " or "))
        nxt = _clause(rng) if rng.random() < 0.7 else _decor(rng)
        if sep == " (":
            parts.append(" (" + nxt + ")")
        else:
            parts.append(sep + nxt)
    s = "".join(parts)
    # quotes
    if rng.random() < 0.18:
        q = rng.choice((('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")))
        s = "%s said, %s%s%s" % (name(rng).split(",")[0], q[0], s, q[1]) if rng.random() < 0.5 \
            else s + ", %s%s%s" % (q[0], " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 4))), q[1])
    # decorations inserted at word boundaries
    if rng.random() < 0.45:
        words = s.split(" ")
        for _ in range(rng.randint(1, 2)):
            words.insert(rng.randint(0, len(words)), _decor(rng))
        s = " ".join(words)
    if rng.random() < 0.12:
        s = rng.choice(_INTERJ) + rng.choice((", ", "! ", " -- ", "... ")) + s
    s = s[:1].upper() + s[1:]
    s += rng.choices((".", "!", "?", "", "...", "!!", "!!!", "?!", " :)", "."), (40, 10, 10, 8, 5, 4, 3, 2, 2, 16))[0]
    return s


def english_text(rng: random.Random, max_sentences: int = 4) -> str:
    n = rng.randint(1, max_sentences)
    out = english_sentence(rng)
    for _ in range(n - 1):
        out += rng.choice((" ", " ", " ", "  ", "\n", "\n\n", "\r\n")) + english_sentence(rng)
    return out


def title_line(rng: random.Random) -> str:
    """Titles and credit lines with & and ; between names."""
    def pair():
        return "%s & %s" % (rng.choice(FIRST_NAMES + LAST_NAMES), rng.choice(FIRST_NAMES + LAST_NAMES))
    r = rng.random()
    if r < 0.3:
        return "; ".join(pair() for _ in range(rng.randint(2, 4)))
    if r < 0.6:
        return "%s %s & %s %s" % (rng.choice(ADJS).capitalize(), rng.choice(NOUNS).capitalize(),
                                  rng.choice(ADJS).capitalize(), _plural(rng.choice(NOUNS)).capitalize())
    if r < 0.8:
        return "%s: %s & %s; %s" % (rng.choice(("Starring", "Hosts", "Authors", "Featuring", "Speakers")),
                                    name(rng), name(rng).split(",")[0], name(rng).split(",")[0])
    return "%s & %s %s, %s" % (rng.choice(LAST_NAMES), rng.choice(LAST_NAMES),
                               rng.choice(("LLP", "Ltd.", "Inc.", "GmbH", "Associates")), rng.choice(("London", "Boston", "Berlin", "Toronto")))


# ==========================================================================
# Multi-script text (composed from inventories)
# ==========================================================================

def _hangul_syllable(rng: random.Random) -> str:
    l = rng.choice((0, 0, 2, 3, 5, 6, 7, 9, 11, 11, 12, 14, 15, 16, 17, 18, 1, 4, 8, 10, 13))
    v = rng.choice((0, 0, 4, 4, 8, 8, 13, 13, 18, 20, 20, 1, 2, 5, 6, 11, 12, 14, 17, 19))
    t = 0 if rng.random() < 0.55 else rng.choice((1, 4, 4, 8, 16, 17, 19, 21, 21, 7, 27, 22))
    return chr(0xAC00 + (l * 21 + v) * 28 + t)


KO_WORDS = _w("""
사람 시간 오늘 내일 어제 학교 회사 친구 가족 음식 주문 배송 가격 상품 문의 확인 서비스
고객 회원 비밀번호 이메일 주소 전화 날씨 사랑 영화 음악 여행 사진 생각 문제 정보 결과
방법 경우 세계 나라 도시 집 물 책 컴퓨터 프로그램 개발 디자인 감사 지금 처음 마지막
다음 이번 정말 아주 조금 많이 같이 다시 빨리 우리 저희 여기 거기 무료 할인 쿠폰 결제
환불 교환 리뷰 후기 추천 이벤트 공지 예약 시작 종료 매장 온라인 앱 버전 업데이트
""")
KO_PARTICLES = ("은", "는", "이", "가", "을", "를", "에", "에서", "의", "도", "로", "와", "과", "하고", "까지", "부터", "만")
KO_ENDINGS = ("합니다", "했습니다", "해요", "있어요", "없어요", "좋아요", "갑니다", "봤어요", "주세요",
              "입니다", "인가요", "할게요", "됩니다", "했어요", "하세요", "드립니다", "같아요")

JA_HIRA = [chr(c) for c in range(0x3042, 0x3094) if chr(c) not in "ぃぅぇぉゃゅょゎっ"]
JA_KANA_WORDS = _w("""
コーヒー パソコン メール サービス アプリ ログイン アカウント パスワード ショップ カート
ポイント キャンペーン クーポン レビュー ニュース ホテル レストラン メニュー チケット
カメラ スマホ ゲーム アニメ ドラマ ミュージック サイズ カラー ブラック ホワイト セール
""")
JA_KANJI_WORDS = _w("""
日本 東京 大阪 会社 時間 今日 明日 昨日 電話 注文 確認 商品 価格 送料 無料 問題 情報
結果 方法 世界 学校 先生 学生 仕事 天気 映画 音楽 旅行 写真 料理 家族 友達 使用 登録
会員 住所 名前 番号 予約 配送 発送 返品 交換 支払 決済 在庫 発売 新作 限定 店舗 地図
駅 電車 新幹線 空港 季節 春 夏 秋 冬 週末 午前 午後 毎日 本当 少し 全部 最近 最初 最後
""")
JA_PARTICLES = ("は", "が", "を", "に", "で", "と", "の", "も", "へ", "から", "まで", "より")
JA_ENDINGS = ("します", "しました", "です", "でした", "ください", "ません", "ですか", "ましょう",
              "でしょう", "ています", "いたします", "だよ", "かな", "ね", "よ")

HANZI = ("的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她"
         "里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从"
         "动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几"
         "很业最间新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太"
         "水万市眼体别处总才场师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放"
         "做系计或司利受光王果亲界及今京务制解各任至清物台象记边共风战干接它许八特觉望直服毛林题建南度统色字请"
         "交爱让认算论百吃义科怎元社术结六功指思非流每青管夫连远资队跟带花快条院变联言权往展该领传近留红治决周"
         "保达办运武半候七必城父强步完革深区即求品士转量空甚众技轻程告江语英基派满式李息写呢识极令黄德收脸钱党"
         "倒未持取设始版双历越史商千片容研像找友孩站广改议形委早房音火际则首单据导影失拿网香似斯专石若兵弟谁校"
         "读志飞观争究包组造落视济喜离虽坏兴切送订购物价格客户服务址邮箱登录注册支付货退款评优惠活免费递")
ZH_WORDS = _w("""
我们 今天 明天 昨天 公司 时间 问题 订单 价格 商品 客户 服务 地址 电话 邮箱 信息 系统 用户
登录 注册 支付 发货 退货 中国 北京 上海 深圳 学习 工作 生活 朋友 家人 天气 电影 音乐 旅游
照片 手机 电脑 网络 软件 开发 设计 产品 质量 非常 可以 没有 已经 现在 因为 所以 但是 如果
这个 那个 什么 怎么 为什么 谢谢 你好 欢迎 帮助 快递 包邮 优惠券 好评 推荐 客服 售后
""")
CJK_PUNCT_MID = ("，", "，", "，", "、", "；", "：")
CJK_PUNCT_END = ("。", "。", "。", "！", "？", "！！", "？！", "…", "～")
CJK_BRACKETS = (("「", "」"), ("『", "』"), ("（", "）"), ("《", "》"), ("【", "】"))

RU_WORDS = _w("""
привет спасибо пожалуйста здравствуйте да нет хорошо плохо сегодня завтра вчера
сейчас время день ночь утро вечер неделя месяц год человек люди друг семья мама
папа дом квартира город страна улица работа компания магазин заказ доставка цена
товар покупка оплата карта номер телефон адрес почта письмо сообщение вопрос ответ
проблема решение помощь поддержка клиент сервис система пользователь пароль вход
регистрация сайт страница программа приложение новый старый большой маленький
красивый быстрый медленный дорогой дешёвый хороший интересный важный простой
сложный первый последний следующий русский английский язык книга фильм музыка
игра погода солнце дождь снег море река лес парк школа университет студент
учитель врач больница машина поезд самолёт билет путешествие отпуск праздник
подарок еда вода чай кофе хлеб молоко мясо рыба суп обед ужин завтрак можно
нужно надо очень много мало всегда никогда иногда здесь там где когда почему как
что кто который этот тот мой твой наш ваш его её их я ты он она мы вы они быть
идти ехать делать сказать знать думать хотеть мочь видеть смотреть читать писать
говорить работать жить любить получить отправить купить заказать проверить
подтвердить ждать и в на с по для от до из у о к за без при дякую будь ласка
привіт їжа місто країна ґанок
""")
EL_WORDS = _w("""
καλημέρα ευχαριστώ παρακαλώ ναι όχι σήμερα αύριο χθες τώρα ώρα μέρα νύχτα πρωί
βράδυ εβδομάδα μήνας χρόνος άνθρωπος φίλος οικογένεια σπίτι πόλη χώρα δρόμος
δουλειά εταιρεία κατάστημα παραγγελία αποστολή τιμή προϊόν πληρωμή κάρτα αριθμός
τηλέφωνο διεύθυνση μήνυμα ερώτηση απάντηση πρόβλημα βοήθεια πελάτης υπηρεσία
σύστημα χρήστης κωδικός σελίδα εφαρμογή νέος παλιός μεγάλος μικρός ωραίος
γρήγορος καλός σημαντικός απλός πρώτος τελευταίος βιβλίο ταινία μουσική καιρός
ήλιος βροχή θάλασσα βουνό σχολείο πανεπιστήμιο γιατρός αυτοκίνητο εισιτήριο ταξίδι
διακοπές φαγητό νερό καφές ψωμί είμαι έχω θέλω μπορώ ξέρω βλέπω διαβάζω γράφω
μιλάω δουλεύω αγαπώ στέλνω αγοράζω και ή αλλά για με σε από το η ο τα οι πολύ
λίγο πάντα ποτέ εδώ εκεί πού πότε γιατί πώς τι
""")
AR_WORDS = _w("""
مرحبا شكرا من فضلك نعم لا اليوم غدا أمس الآن وقت يوم ليلة صباح مساء أسبوع شهر
سنة إنسان صديق عائلة بيت مدينة بلد شارع عمل شركة متجر طلب شحن سعر منتج دفع
بطاقة رقم هاتف عنوان بريد رسالة سؤال جواب مشكلة مساعدة عميل خدمة نظام مستخدم
كلمة صفحة تطبيق جديد قديم كبير صغير جميل سريع جيد مهم بسيط أول آخر كتاب فيلم
موسيقى طقس شمس مطر بحر جبل مدرسة جامعة طبيب سيارة تذكرة سفر عطلة طعام ماء قهوة
خبز أنا أنت هو هي نحن هم في على إلى مع عن هذا هذه ذلك الذي التي كان يكون أريد
يمكن أعرف أرى أقرأ أكتب أتكلم أعمل أحب أرسل أشتري و أو لكن جدا قليلا دائما
هنا هناك أين متى لماذا كيف ماذا السلام عليكم الحمد لله
""")
HE_WORDS = _w("""
שלום תודה בבקשה כן לא היום מחר אתמול עכשיו זמן יום לילה בוקר ערב שבוע חודש שנה
אדם חבר משפחה בית עיר ארץ רחוב עבודה חברה חנות הזמנה משלוח מחיר מוצר תשלום
כרטיס מספר טלפון כתובת דואר הודעה שאלה תשובה בעיה עזרה לקוח שירות מערכת משתמש
סיסמה דף אפליקציה חדש ישן גדול קטן יפה מהיר טוב חשוב פשוט ראשון אחרון ספר סרט
מוזיקה מזג אוויר שמש גשם ים הר רופא מכונית טיול חופשה אוכל מים קפה לחם אני
אתה הוא היא אנחנו הם של על אל עם זה זאת רוצה יכול יודע רואה קורא כותב מדבר
עובד אוהב שולח קונה או אבל מאוד קצת תמיד כאן שם איפה מתי למה איך מה
""")
TH_WORDS = _w("""
สวัสดี ขอบคุณ ครับ ค่ะ ใช่ ไม่ วันนี้ พรุ่งนี้ เมื่อวาน ตอนนี้ เวลา วัน คืน เช้า เย็น สัปดาห์
เดือน ปี คน เพื่อน ครอบครัว บ้าน เมือง ประเทศ ถนน งาน บริษัท ร้าน สั่งซื้อ จัดส่ง ราคา
สินค้า ชำระเงิน บัตร หมายเลข โทรศัพท์ ที่อยู่ อีเมล ข้อความ คำถาม คำตอบ ปัญหา ช่วย ลูกค้า
บริการ ระบบ ผู้ใช้ รหัสผ่าน หน้า แอป ใหม่ เก่า ใหญ่ เล็ก สวย เร็ว ดี สำคัญ ง่าย แรก สุดท้าย
หนังสือ หนัง เพลง อากาศ ฝน ทะเล ภูเขา โรงเรียน มหาวิทยาลัย หมอ รถ ตั๋ว เที่ยว วันหยุด
อาหาร น้ำ กาแฟ ขนมปัง ผม ฉัน คุณ เขา เรา ของ ที่ ใน และ หรือ แต่ มาก นิดหน่อย เสมอ
ที่นี่ ที่นั่น ที่ไหน เมื่อไหร่ ทำไม อย่างไร อะไร อยาก ได้ รู้ เห็น อ่าน เขียน พูด ทำงาน รัก ส่ง ซื้อ
""")
HI_WORDS = _w("""
नमस्ते धन्यवाद कृपया हाँ नहीं आज कल अभी समय दिन रात सुबह शाम सप्ताह महीना साल
आदमी लोग दोस्त परिवार घर शहर देश सड़क काम कंपनी दुकान ऑर्डर डिलीवरी कीमत सामान
भुगतान कार्ड नंबर फ़ोन पता ईमेल संदेश सवाल जवाब समस्या मदद ग्राहक सेवा सिस्टम
पासवर्ड पेज ऐप नया पुराना बड़ा छोटा सुंदर तेज़ अच्छा ज़रूरी आसान पहला आखिरी किताब
फ़िल्म संगीत मौसम सूरज बारिश समुद्र पहाड़ स्कूल डॉक्टर गाड़ी टिकट यात्रा छुट्टी
खाना पानी चाय कॉफ़ी रोटी मैं तुम आप वह हम वे का की के में पर से को और या लेकिन
बहुत थोड़ा हमेशा कभी यहाँ वहाँ कहाँ कब क्यों कैसे क्या है हैं था थी चाहता सकता
जानता देखता पढ़ता लिखता बोलता करता प्यार भेजना खरीदना
""")
VI_WORDS = _w("""
xin chào cảm ơn vui lòng có không hôm nay ngày mai qua bây giờ thời gian ngày đêm
buổi sáng tối tuần tháng năm người bạn gia đình nhà thành phố đất nước đường công
việc ty cửa hàng đơn giao giá sản phẩm thanh toán thẻ số điện thoại địa chỉ thư tin
nhắn câu hỏi trả lời vấn đề giúp đỡ khách dịch vụ hệ thống dùng mật khẩu trang ứng
dụng mới cũ lớn nhỏ đẹp nhanh tốt quan trọng đơn giản đầu tiên cuối cùng sách phim
âm nhạc thời tiết mặt trời mưa biển núi trường học đại bác sĩ xe vé du lịch kỳ
nghỉ đồ ăn nước cà phê bánh mì phở tôi anh chị em chúng họ của trong và hoặc nhưng
rất một chút luôn bao giờ ở đây đó đâu khi nào tại sao như thế cái gì muốn được
biết thấy đọc viết nói làm yêu gửi mua
""")
LATIN_ACCENT_WORDS = _w("""
café crème éléphant à où été déjà très après père mère frère sœur garçon façade
naïve résumé élève bientôt hôtel forêt fenêtre château théâtre île août noël
préféré numéro téléphone adresse commande livraison prix réduction reçu merci
beaucoup bonjour demain aujourd'hui s'il plaît niño señor año mañana corazón
canción jalapeño piñata información teléfono dirección pedido envío precio también
aquí allí está están más qué cómo dónde cuándo gracias favor jamón árbol lápiz
ratón camión médico música película fácil difícil útil Grüße schön Straße Mädchen
Bäcker Größe Übung über für müssen können Käse Brötchen Frühstück Bestellung
Lieferung Preis Rückgabe Tschüss bitte danke morgen heute Fußball Schlüssel não
ação coração informação atenção pão mãe irmão avô avó você obrigado obrigada até
amanhã endereço preço entrega São perché città caffè però più così già università
lunedì venerdì Łódź źle żółw gęś pięć cześć dziękuję proszę jutro dzisiaj
zamówienie dostawa więcej děkuji prosím řeka čeština škola město příští zítra dnes
objednávka teşekkürler lütfen günaydın çok güzel şehir İstanbul ağaç öğrenci
sipariş ücretsiz kargo fiyat ödeme smørbrød søster blåbær København Göteborg Malmö
Århus tusen takk være själv även mulțumesc București țară köszönöm szép tűz idő
""")
_ACC_V = "aáàâäãåeéèêëiíìîïoóòôöõøuúùûüyý"
_ACC_C = "bcçdfghjklmnñprsštvzžłř"

_EMOJI_RANGES = ((0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6C5), (0x1F90C, 0x1F9FF),
                 (0x2600, 0x26FF), (0x2700, 0x27BF))
_FLAGS = ("US", "GB", "JP", "KR", "CN", "DE", "FR", "ES", "IT", "BR", "IN", "CA", "AU", "MX", "UA", "SE")


def emoji(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.08:
        cc = rng.choice(_FLAGS)
        return "".join(chr(0x1F1E6 + ord(c) - 65) for c in cc)
    if r < 0.14:
        return rng.choice(("\U0001F468\u200D\U0001F469\u200D\U0001F467", "\U0001F469\u200D\U0001F4BB",
                           "\U0001F3F3\uFE0F\u200D\U0001F308", "\u2764\uFE0F\u200D\U0001F525"))
    lo, hi = rng.choice(_EMOJI_RANGES)
    e = chr(rng.randint(lo, hi))
    if rng.random() < 0.1 and lo == 0x1F300:
        e += chr(rng.randint(0x1F3FB, 0x1F3FF))
    elif rng.random() < 0.15 and lo in (0x2600, 0x2700):
        e += "\uFE0F"
    return e * rng.choice((1, 1, 1, 1, 2, 3))


def _ascii_mix(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.25:
        return rng.choice(("iPhone", "Android", "Wi-Fi", "PDF", "USB-C", "API", "OK", "App", "Excel",
                           "Zoom", "QR", "SNS", "PC", "TV", "AI", "SALE", "VIP", "GPS", "LTE", "5G"))
    if r < 0.45:
        return str(rng.choice((rng.randint(1, 100), rng.randint(100, 99999))))
    if r < 0.55:
        return url(rng)
    if r < 0.6:
        return email(rng)
    if r < 0.7:
        return money(rng)
    if r < 0.78:
        return percent(rng)
    if r < 0.85:
        return clock(rng)
    if r < 0.92:
        return emoji(rng)
    return rng.choice(VOCAB)


def _script_word(rng: random.Random, script: str) -> str:
    if script == "ko":
        if rng.random() < 0.5:
            w = rng.choice(KO_WORDS)
        else:
            w = "".join(_hangul_syllable(rng) for _ in range(rng.randint(1, 3)))
        if rng.random() < 0.5:
            w += rng.choice(KO_PARTICLES)
        return w
    if script == "ja":
        r = rng.random()
        if r < 0.35:
            w = rng.choice(JA_KANJI_WORDS)
        elif r < 0.5:
            w = rng.choice(JA_KANA_WORDS)
        elif r < 0.7:
            w = "".join(rng.choice(HANZI) for _ in range(rng.randint(1, 2))) + \
                "".join(rng.choice(JA_HIRA) for _ in range(rng.randint(1, 3)))
        else:
            w = "".join(rng.choice(JA_HIRA) for _ in range(rng.randint(2, 5)))
        if rng.random() < 0.5:
            w += rng.choice(JA_PARTICLES)
        return w
    if script == "zh":
        if rng.random() < 0.45:
            return rng.choice(ZH_WORDS)
        return "".join(rng.choice(HANZI) for _ in range(rng.randint(1, 3)))
    if script == "ru":
        return rng.choice(RU_WORDS)
    if script == "el":
        return rng.choice(EL_WORDS)
    if script == "ar":
        return rng.choice(AR_WORDS)
    if script == "he":
        return rng.choice(HE_WORDS)
    if script == "th":
        return rng.choice(TH_WORDS)
    if script == "hi":
        return rng.choice(HI_WORDS)
    if script == "vi":
        return rng.choice(VI_WORDS)
    # accented latin
    if rng.random() < 0.7:
        return rng.choice(LATIN_ACCENT_WORDS)
    return "".join(rng.choice(_ACC_C) + rng.choice(_ACC_V) for _ in range(rng.randint(1, 4)))


SCRIPTS = ("ko", "ja", "zh", "ru", "el", "ar", "he", "th", "hi", "vi", "latin")


def script_text(rng: random.Random, script: Optional[str] = None) -> str:
    """Generated text in one script, mixed with ASCII tokens."""
    if script is None:
        script = rng.choice(SCRIPTS + ("zh", "ja", "ko", "ru"))
    cjk = script in ("zh", "ja")
    n = rng.randint(2, 18)
    out = []
    for i in range(n):
        if rng.random() < 0.13:
            tok = _ascii_mix(rng)
        else:
            tok = _script_word(rng, script)
        if cjk and rng.random() < 0.08:
            a, b = rng.choice(CJK_BRACKETS)
            tok = a + tok + b
        out.append(tok)
        if i < n - 1:
            if cjk:
                sep = rng.choice(CJK_PUNCT_MID) if rng.random() < 0.18 else ("" if rng.random() < 0.85 else " ")
            elif script == "th":
                sep = " " if rng.random() < 0.3 else ""
            elif script == "ar":
                sep = rng.choice(("، ", " ")) if rng.random() < 0.15 else " "
            elif script == "hi":
                sep = ", " if rng.random() < 0.1 else " "
            else:
                sep = ", " if rng.random() < 0.12 else (" - " if rng.random() < 0.02 else " ")
            out.append(sep)
    s = "".join(out)
    if script == "ko" and rng.random() < 0.6:
        s += " " + rng.choice(KO_ENDINGS)
    if script == "ja" and rng.random() < 0.6:
        s += rng.choice(JA_ENDINGS)
    if cjk:
        s += rng.choice(CJK_PUNCT_END) if rng.random() < 0.85 else ""
    elif script == "ar":
        s += rng.choice((".", "؟", "!", ""))
    elif script == "hi":
        s += rng.choice(("।", "।", "?", "!", ""))
    elif script == "el":
        s += rng.choice((".", ";", "!", ""))  # Greek question mark is ';'
    elif script == "th":
        s += rng.choice(("", "", " ครับ", " ค่ะ", "!"))
    else:
        s += rng.choice((".", ".", "!", "?", "", "..."))
    if s and script in ("latin", "ru", "el", "vi"):
        s = s[0].upper() + s[1:]
    if rng.random() < 0.15:
        s += " " + emoji(rng)
    return s


def mixed_text(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return script_text(rng)
    if r < 0.6:
        return script_text(rng) + rng.choice((" ", "\n", " / ", " | ", " - ")) + script_text(rng)
    if r < 0.8:
        return english_sentence(rng) + rng.choice((" ", " (", " / ", "\n")) + script_text(rng)
    return script_text(rng) + " " + english_sentence(rng)


# ==========================================================================
# Technical text
# ==========================================================================

def _nm(rng: random.Random) -> str:
    return rng.choice(IDENT_WORDS)


def _camel(rng: random.Random) -> str:
    return _nm(rng) + _nm(rng).capitalize()


def code_line(rng: random.Random) -> str:
    a, b, c = _nm(rng), _nm(rng), _nm(rng)
    n = rng.randint(0, 1000)
    op = rng.choice(("+", "-", "*", "/", "%"))
    cmp_ = rng.choice(("<", ">", "<=", ">=", "==", "!="))
    cls = rng.choice(IDENT_WORDS) + "-" + rng.choice(IDENT_WORDS)
    Name = _nm(rng).capitalize() + _nm(rng).capitalize()
    lang = rng.choice(("js", "js", "py", "py", "c", "java", "go", "rust", "css", "ts", "sql_free", "ruby"))
    if lang in ("js", "ts"):
        return rng.choice((
            "function %s(%s, %s) { return %s %s %s; }" % (_camel(rng), a, b, a, op, b),
            "const %s = %s.map((x) => x * %d);" % (_camel(rng), a, n),
            "if (%s && %s) { %s++; }" % (a, b, c),
            "if (%s.length %s %d || !%s) return;" % (a, cmp_, n, b),
            "let %s = %s ?? %d;" % (a, b, n),
            "for (let i = 0; i < %s.length; i++) { total += %s[i]; }" % (a, a),
            "export default function %s() { return null; }" % Name,
            "console.log('%s', %s);" % (rng.choice(VOCAB), a),
            "document.querySelector('.%s').classList.add('is-%s');" % (cls, rng.choice(ADJS)),
            "fetch('/api/%s/' + id).then((r) => r.json());" % a,
            "const { %s, %s } = %s;" % (a, b, c),
            "await %s.%s(%d);" % (a, _camel(rng), n),
            "%s.filter((x) => x !== null).length %s %d" % (a, cmp_, n),
            "setTimeout(() => %s(), %d);" % (_camel(rng), n),
            "import { %s } from './%s';" % (Name, a),
            "const %s: number[] = [%d, %d, %d];" % (a, n, n + 1, n * 2),
            "interface %s { %s: string; %s?: number }" % (Name, a, b),
            "x = (a + %d) * b %% %d;" % (n, rng.randint(2, 9)),
        ))
    if lang == "py":
        return rng.choice((
            "def %s(%s, %s=None): return %s if %s is None else %s %s %s" % (a, b, c, b, c, b, op, c),
            "for %s in range(%d): print(%s)" % (a[0], n, a[0]),
            "%s = [x for x in %s if x %s %d]" % (a, b, cmp_, n),
            "with open('%s') as f: data = f.read()" % filename(rng),
            "from %s import %s" % (a, Name),
            "if __name__ == '__main__': main()",
            "%s = {'%s': %d, '%s': '%s'}" % (a, b, n, c, rng.choice(VOCAB)),
            "return sum(x ** 2 for x in %s)" % a,
            "class %s(%s): pass" % (Name, rng.choice(("object", "Exception", "BaseModel", "dict"))),
            "log.info('loaded %%d rows from %%s', len(%s), path)" % a,
            "%s = %s.get('%s', %d)" % (a, b, c, n),
            "assert %s %s %d, 'unexpected %s'" % (a, cmp_, n, b),
            "lambda x: x %s %d" % (op, rng.randint(1, 9)),
        ))
    if lang == "c":
        return rng.choice((
            "int %s(int %s, int %s) { return %s %s %s; }" % (a, b, c, b, op, c),
            "for (int i = 0; i < %d; i++) { sum += %s[i]; }" % (n, a),
            "printf(\"%%d %s\\n\", %s);" % (b, a),
            "#include <%s.h>" % rng.choice(("stdio", "stdlib", "string", "stdint", "math", "assert")),
            "if (%s == NULL) return -1;" % a,
            "char *%s = malloc(%d * sizeof(char));" % (a, n),
            "#define MAX_%s %d" % (a.upper(), n),
            "static const uint32_t %s = 0x%s;" % (a.upper(), rand_hex(rng, 8)),
            "while (%s %s %d) { %s--; }" % (a, cmp_, n, a),
        ))
    if lang == "java":
        return rng.choice((
            "public static int %s(int %s) { return %s * %d; }" % (_camel(rng), a, a, n),
            "List<String> %s = new ArrayList<>();" % _camel(rng),
            "Map<String, Integer> %s = new HashMap<>();" % _camel(rng),
            "System.out.println(\"%s: \" + %s);" % (rng.choice(VOCAB), a),
            "@Override public String toString() { return \"%s\"; }" % Name,
            "if (%s != null && !%s.isEmpty()) { process(%s); }" % (a, a, a),
            "private final %s %s;" % (Name, a),
        ))
    if lang == "go":
        return rng.choice((
            "func %s(%s int) int { return %s * %d }" % (a, b, b, n),
            "if err != nil { return err }",
            "%s := make([]string, 0, %d)" % (a, n),
            "fmt.Printf(\"%%s: %%d\\n\", %s, %s)" % (a, b),
            "for _, v := range %s { total += v }" % a,
            "type %s struct { %s string }" % (Name, a.capitalize()),
        ))
    if lang == "rust":
        return rng.choice((
            "fn %s(%s: u32) -> u32 { %s * %d }" % (a, b, b, n),
            "let %s: Vec<u32> = (0..%d).collect();" % (a, n),
            "match %s { Some(v) => v, None => %d }" % (a, n),
            "println!(\"{} %s\", %s);" % (rng.choice(NOUNS), a),
            "#[derive(Debug, Clone, PartialEq)]",
            "impl %s { pub fn new() -> Self { Self { %s: 0 } } }" % (Name, a),
            "let %s = %s.iter().filter(|x| **x %s %d).count();" % (a, b, cmp_, n),
        ))
    if lang == "css":
        return rng.choice((
            ".%s { color: #%s; margin: %dpx auto; }" % (cls, rand_hex(rng, rng.choice((3, 6))), rng.randint(0, 40)),
            "@media (max-width: %dpx) { .%s { display: none; } }" % (rng.choice((480, 600, 768, 1024)), cls),
            "#%s > .%s:hover { background: rgba(0, 0, 0, 0.%d); }" % (a, cls, rng.randint(1, 9)),
            "font-family: 'Helvetica Neue', Arial, sans-serif;",
            "grid-template-columns: repeat(%d, 1fr); gap: %drem;" % (rng.randint(2, 6), rng.randint(1, 4)),
            "transition: opacity %dms ease-in-out;" % rng.randint(100, 900),
        ))
    if lang == "ruby":
        return rng.choice((
            "%s.each { |x| puts x }" % a,
            "def %s(%s) = %s * %d" % (a, b, b, n),
            "%s = %s.select(&:%s?)" % (a, b, rng.choice(ADJS).replace("-", "_")),
        ))
    # sql_free: plain expressions
    return "%s %s %d %s %s %s %d" % (a, cmp_, n, rng.choice(("and", "&&", "or", "||")), b, cmp_, n + 5)


def arithmetic(rng: random.Random) -> str:
    def num():
        return rng.choice((str(rng.randint(0, 100)), "%.2f" % (rng.random() * 100), rng.choice("xyzabn")))
    ops = ("+", "-", "*", "/", "%", "^", "**", "mod")
    r = rng.random()
    if r < 0.3:
        e = num()
        for _ in range(rng.randint(1, 4)):
            e += " %s %s" % (rng.choice(ops), num())
        return e + rng.choice((" = %s" % num(), "", ";", " ≈ %s" % num()))
    if r < 0.5:
        return "%s %s %s %s %s %s %s" % (num(), rng.choice(("<", ">", "<=", ">=", "≤", "≥")), num(),
                                         rng.choice(("and", "&&", "or", "||", ",", ";")), num(),
                                         rng.choice(("<", ">", "==", "!=")), num())
    if r < 0.65:
        return "%d %% %d = %d; %d %% %d = %d" % tuple(rng.randint(1, 50) for _ in range(6))
    if r < 0.8:
        return "f(x) = %dx^2 + %dx - %d, x > %d" % (rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9), rng.randint(0, 5))
    return "%s%% of %s = %s; up %s%% from %s" % (rng.randint(1, 99), money(rng), money(rng), rng.randint(1, 40), rng.choice(_MONTHS))


def config_line(rng: random.Random) -> str:
    k, s = _nm(rng), _nm(rng)
    v = rng.choice((str(rng.randint(0, 65535)), "true", "false", rng.choice(IDENT_WORDS), neutral_host(rng),
                    "%d.%d.%d" % (rng.randint(0, 9), rng.randint(0, 30), rng.randint(0, 50)), date_time(rng),
                    "/var/lib/" + k, "%dms" % rng.randint(10, 9000), "%dMB" % rng.randint(1, 4096)))
    return rng.choice((
        "[%s]" % s, "[%s.%s]" % (s, k), "%s = %s" % (k, v), "%s=%s" % (k, v), "; %s" % english_sentence(rng),
        "%s: %s" % (k, v), "%s: \"%s\"" % (k, v), "- %s" % v, "%s = \"%s\"" % (k, v),
        "%s = [%d, %d, %d]" % (k, rng.randint(1, 9), rng.randint(10, 99), rng.randint(100, 999)),
        "%s.%s.%s=%s" % (s, k, _nm(rng), v), "%s_%s=%s" % (s.upper(), k.upper(), v),
        "# %s" % english_sentence(rng), "%s:\n  %s: %s\n  %s: %s" % (s, k, v, _nm(rng), rng.randint(1, 99)),
    ))


def regex(rng: random.Random) -> str:
    pieces = (r"\d+", r"\d{%d}" % rng.randint(2, 6), r"[a-z]+", r"[A-Za-z0-9_]*", r"\s*", r"\w+", r"(?:%s|%s)" % (_nm(rng), _nm(rng)),
              r"[^/]+", r"\.", r"-", r"_", r"[0-9a-f]{%d}" % rng.choice((8, 32, 40)), r"(\d{4})-(\d{2})-(\d{2})", r".*?",
              r"\b%s\b" % _nm(rng), r"[\u4e00-\u9fff]+", r"[+-]?\d+(\.\d+)?", r"(?P<%s>\w+)" % _nm(rng))
    body = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 5)))
    if rng.random() < 0.5:
        body = "^" + body + "$"
    r = rng.random()
    if r < 0.3:
        return "/%s/%s" % (body, rng.choice(("", "g", "i", "gi", "m")))
    if r < 0.5:
        return "re.compile(r'%s')" % body
    if r < 0.7:
        return "pattern: %s" % body
    return body


def version_constraint(rng: random.Random) -> str:
    def v():
        return "%d.%d.%d" % (rng.randint(0, 9), rng.randint(0, 30), rng.randint(0, 40))
    return rng.choice((
        ">=%s,<%d.0" % (v(), rng.randint(2, 10)), "^%s" % v(), "~%s" % v(), "~=%d.%d" % (rng.randint(1, 4), rng.randint(0, 12)),
        "%d.%d.x" % (rng.randint(0, 9), rng.randint(0, 20)), "^%s || ^%s" % (v(), v()), ">= %s < %s" % (v(), v()),
        "[%d.0,%d.0)" % (rng.randint(1, 3), rng.randint(4, 6)), "%s-beta.%d+build.%d" % (v(), rng.randint(1, 9), rng.randint(1, 999)),
        "v%s-rc.%d" % (v(), rng.randint(1, 5)), "%s == %s" % (_nm(rng), v()), "%s>=%s" % (_nm(rng), v()),
        "%s..%s" % (v(), v()), "versions %s...%s" % (v(), v()),
        "\"%s\": \"^%s\"" % (_nm(rng), v()),
    ))


_LANGS = ("en-US", "en-GB", "en", "fr-FR", "fr", "de-DE", "de", "es-ES", "es", "ja-JP", "ja", "ko-KR", "ko",
          "zh-CN", "zh-TW", "zh", "ru-RU", "ru", "pt-BR", "it", "nl", "sv", "pl", "tr", "ar", "he", "hi", "th", "vi")


def user_agent(rng: random.Random) -> str:
    cv = "%d.0.%d.%d" % (rng.randint(90, 131), rng.randint(1000, 6999), rng.randint(0, 250))
    plat = rng.choice(("Windows NT 10.0; Win64; x64", "Macintosh; Intel Mac OS X 10_15_7", "X11; Linux x86_64",
                       "X11; Ubuntu; Linux x86_64", "Windows NT 6.1; WOW64"))
    r = rng.random()
    if r < 0.35:
        return "Mozilla/5.0 (%s) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/%s Safari/537.36" % (plat, cv) + \
            (" Edg/%s" % cv if rng.random() < 0.15 else "")
    if r < 0.5:
        fv = rng.randint(90, 132)
        return "Mozilla/5.0 (%s; rv:%d.0) Gecko/20100101 Firefox/%d.0" % (plat, fv, fv)
    if r < 0.65:
        iv = "%d_%d" % (rng.randint(14, 18), rng.randint(0, 6))
        dev = rng.choice(("iPhone; CPU iPhone OS %s like Mac OS X" % iv, "iPad; CPU OS %s like Mac OS X" % iv))
        return "Mozilla/5.0 (%s) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/%d.%d Mobile/15E148 Safari/604.1" % (
            dev, rng.randint(14, 18), rng.randint(0, 6))
    if r < 0.78:
        return "Mozilla/5.0 (Linux; Android %d; %s) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/%s Mobile Safari/537.36" % (
            rng.randint(9, 15), rng.choice(("Pixel 7", "Pixel 8 Pro", "SM-S918B", "SM-A546E", "M2101K6G", "CPH2451", "K")), cv)
    if r < 0.88:
        return "%s/%d.%d.%d (%s; %s %d.%d; Scale/%d.00)" % (
            brand(rng).capitalize(), rng.randint(1, 9), rng.randint(0, 30), rng.randint(0, 99),
            rng.choice(("iPhone", "iPad", "Android")), rng.choice(("iOS", "Android")), rng.randint(9, 18),
            rng.randint(0, 6), rng.randint(2, 3))
    return rng.choice(("curl/%d.%d.%d" % (rng.randint(7, 8), rng.randint(0, 80), rng.randint(0, 9)),
                       "python-requests/2.%d.%d" % (rng.randint(20, 32), rng.randint(0, 9)),
                       "okhttp/4.%d.%d" % (rng.randint(0, 12), rng.randint(0, 9)), "Go-http-client/2.0",
                       "PostmanRuntime/7.%d.%d" % (rng.randint(20, 40), rng.randint(0, 9)),
                       "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
                       "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
                       "axios/1.%d.%d" % (rng.randint(0, 7), rng.randint(0, 9))))


MIME_TYPES = _w("""
text/html text/plain text/css text/csv text/javascript text/markdown application/json
application/xml application/pdf application/zip application/gzip application/octet-stream
application/x-www-form-urlencoded application/ld+json application/vnd.api+json
application/vnd.ms-excel application/vnd.openxmlformats-officedocument.wordprocessingml.document
application/msword application/rtf image/png image/jpeg image/gif image/webp image/avif
image/svg+xml audio/mpeg audio/ogg video/mp4 video/webm font/woff2 multipart/form-data
multipart/mixed application/grpc application/x-ndjson application/problem+json
""")


def mime_type(rng: random.Random) -> str:
    m = rng.choice(MIME_TYPES)
    if m.startswith("text/") or m == "application/json":
        if rng.random() < 0.5:
            m += rng.choice(("; charset=utf-8", "; charset=UTF-8", ";charset=iso-8859-1"))
    if m.startswith("multipart/"):
        m += "; boundary=" + rng.choice(("----WebKitFormBoundary", "----", "--", "")) + rand_alnum(rng, 16)
    return m


def header_value(rng: random.Random) -> str:
    r = rng.randint(0, 27)
    if r == 0:
        return user_agent(rng)
    if r == 1:
        items = rng.sample(("text/html", "application/xhtml+xml", "application/xml", "image/avif", "image/webp",
                            "image/apng", "*/*", "application/json", "text/plain"), rng.randint(1, 5))
        return ",".join(i + (";q=0.%d" % rng.randint(1, 9) if rng.random() < 0.3 else "") for i in items)
    if r == 2:
        ls = rng.sample(_LANGS, rng.randint(1, 4))
        return ",".join(l if i == 0 else "%s;q=0.%d" % (l, 9 - i) for i, l in enumerate(ls))
    if r == 3:
        return ", ".join(rng.sample(("gzip", "deflate", "br", "zstd", "identity"), rng.randint(1, 4)))
    if r == 4:
        return rng.choice(("no-cache", "no-store", "max-age=%d" % rng.randint(0, 31536000), "private, max-age=0",
                           "public, max-age=%d, immutable" % rng.randint(60, 31536000), "no-cache, no-store, must-revalidate",
                           "s-maxage=%d, stale-while-revalidate=%d" % (rng.randint(1, 600), rng.randint(1, 600))))
    if r == 5:
        return mime_type(rng)
    if r == 6:
        return "Bearer " + rand_b64(rng, rng.randint(20, 60), url=True)
    if r == 7:
        return rng.choice(('"%s"' % rand_hex(rng, 32), 'W/"%x-%s"' % (rng.randint(100, 99999), rand_b64(rng, 27, True))))
    if r == 8:
        return 'attachment; filename="%s"' % filename(rng)
    if r == 9:
        a = rng.randint(0, 100000)
        return "bytes=%d-%d" % (a, a + rng.randint(1, 1 << 20))
    if r == 10:
        return date_time(rng)
    if r == 11:
        return "%s-%s-%s-%s-%s" % (rand_hex(rng, 8), rand_hex(rng, 4), rand_hex(rng, 4), rand_hex(rng, 4), rand_hex(rng, 12))
    if r == 12:
        return "https://" + neutral_host(rng)
    if r == 13:
        return url(rng)
    if r == 14:
        return ", ".join(ipv4(rng) for _ in range(rng.randint(1, 4)))
    if r == 15:
        return "for=%s;proto=https;by=%s" % (ipv4(rng), ipv4(rng))
    if r == 16:
        return "max-age=%d; includeSubDomains%s" % (rng.choice((31536000, 63072000, 86400)), rng.choice(("", "; preload")))
    if r == 17:
        return "default-src 'self'; img-src 'self' https://%s; style-src 'self' 'unsafe-inline'" % neutral_host(rng)
    if r == 18:
        return rng.choice(("nginx/1.%d.%d" % (rng.randint(18, 27), rng.randint(0, 9)), "Apache/2.4.%d" % rng.randint(40, 62),
                           "cloudflare", "envoy", "Caddy", "gunicorn"))
    if r == 19:
        return rng.choice(("keep-alive", "close", "Upgrade"))
    if r == 20:
        return '"Chromium";v="%d", "Not?A_Brand";v="%d", "Google Chrome";v="%d"' % (rng.randint(110, 131), rng.randint(8, 99), rng.randint(110, 131))
    if r == 21:
        return str(rng.randint(0, 10 ** rng.randint(1, 7)))
    if r == 22:
        return '<%s>; rel="%s"' % (url(rng), rng.choice(("next", "prev", "preload", "canonical", "stylesheet")))
    if r == 23:
        return "1.1 %s (%s)" % (neutral_host(rng), rng.choice(("squid/5.7", "Varnish", "nginx")))
    if r == 24:
        return "Basic " + base64.b64encode(("%s:%s" % (_nm(rng), rand_alnum(rng, 10))).encode()).decode()
    if r == 25:
        return cookie_header(rng)
    if r == 26:
        return rng.choice(("1", "0", "?0", "?1", "same-origin", "cors", "navigate", "document", "empty"))
    return 'form-data; name="%s"; filename="%s"' % (rng.choice(PARAM_NAMES), filename(rng))


def posix_path(rng: random.Random) -> str:
    root = rng.choice(("/home/%s" % rng.choice(FIRST_NAMES).lower(), "/usr/local/bin", "/usr/share", "/opt/%s" % _nm(rng),
                       "/var/lib/%s" % _nm(rng), "/srv/www", "/tmp", "~", ".", "./src", "src", "docs", "/mnt/data",
                       "/Users/%s/Documents" % rng.choice(FIRST_NAMES).lower(), "/var/log/%s" % _nm(rng)))
    segs = [rng.choice(IDENT_WORDS) for _ in range(rng.randint(0, 3))]
    return "/".join([root] + segs + [filename(rng)])


def windows_path(rng: random.Random) -> str:
    root = rng.choice(("C:\\Users\\%s" % rng.choice(FIRST_NAMES), "C:\\Users\\%s\\Documents" % rng.choice(FIRST_NAMES),
                       "D:\\Projects", "C:\\Program Files\\%s" % brand(rng).capitalize(), "E:\\Backup",
                       "C:\\Users\\%s\\Desktop" % rng.choice(FIRST_NAMES), "\\\\%s\\share" % _nm(rng), "C:\\data"))
    segs = [rng.choice(IDENT_WORDS).capitalize() for _ in range(rng.randint(0, 3))]
    return "\\".join([root] + segs + [filename(rng)])


def query_value(rng: random.Random) -> str:
    """A short raw value as it appears in a query string (no encoding needed)."""
    r = rng.random()
    if r < 0.25:
        return str(rng.randint(0, 9999))
    if r < 0.45:
        return rng.choice(IDENT_WORDS)
    if r < 0.55:
        return rng.choice(("true", "false", "1", "0", "on", "off", "yes", "no", "asc", "desc", "all", "none"))
    if r < 0.65:
        return "%d.%d.%d" % (rng.randint(0, 9), rng.randint(0, 20), rng.randint(0, 40))
    if r < 0.72:
        return rand_hex(rng, rng.choice((6, 8, 12, 16, 32)))
    if r < 0.8:
        return "+".join(rng.choice(VOCAB).replace("'", "") for _ in range(rng.randint(2, 4)))
    if r < 0.86:
        return rng.choice(_LANGS)
    if r < 0.92:
        return "%04d-%02d-%02d" % (rng.randint(2000, 2027), rng.randint(1, 12), rng.randint(1, 28))
    return rand_alnum(rng, rng.randint(4, 20))


def query_string(rng: random.Random, n: Optional[int] = None) -> str:
    if n is None:
        n = rng.choices((1, 2, 3, 4, 5, 6, 8, 10), (20, 20, 18, 14, 10, 8, 6, 4))[0]
    short = rng.random() < 0.25
    pairs = []
    for _ in range(n):
        k = rng.choice("abcdefghijklmnopqrstuvwxyz") if short else rng.choice(PARAM_NAMES)
        v = str(rng.randint(0, 99)) if short and rng.random() < 0.7 else query_value(rng)
        pairs.append("%s=%s" % (k, v) if rng.random() > 0.04 else k)
    return "&".join(pairs)


def form_body(rng: random.Random) -> str:
    n = rng.choices((1, 2, 3, 4, 5, 6), (15, 25, 25, 15, 12, 8))[0]
    pairs = []
    for _ in range(n):
        k = rng.choice(PARAM_NAMES)
        r = rng.random()
        if r < 0.3:
            v = english_sentence(rng)[:rng.randint(10, 90)].strip() or "ok"
        elif r < 0.45:
            v = email(rng)
        elif r < 0.55:
            v = name(rng)
        elif r < 0.65:
            v = script_text(rng)
        elif r < 0.72:
            v = money(rng) + " " + percent(rng)
        else:
            v = query_value(rng)
        plus = rng.random() < 0.65
        pairs.append(k + "=" + pct_encode_text(v, plus, upper=rng.random() < 0.8))
    return "&".join(pairs)


def cookie_header(rng: random.Random) -> str:
    n = rng.randint(1, 6)
    out = []
    for _ in range(n):
        k = rng.choice(COOKIE_NAMES)
        v = rng.choice((rand_hex(rng, 32), rand_alnum(rng, rng.randint(16, 40)), rng.choice(IDENT_WORDS),
                        "GA1.2.%d.%d" % (rng.randint(10 ** 8, 10 ** 9), rng.randint(10 ** 9, 2 * 10 ** 9)),
                        rand_b64(rng, rng.randint(20, 60), url=True), str(rng.randint(0, 99999)),
                        rng.choice(_LANGS), "true", "dark", "light",
                        pct_encode_text(rng.choice(("%s|%s" % (_nm(rng), rng.randint(1, 99)), "{\"%s\":%d}" % (_nm(rng), rng.randint(0, 9)))), False)))
        out.append("%s=%s" % (k, v))
    return "; ".join(out)


def csv_row(rng: random.Random, sep: str = ",") -> str:
    cells = []
    for _ in range(rng.randint(3, 9)):
        r = rng.random()
        if r < 0.2:
            c = name(rng)
        elif r < 0.35:
            c = str(rng.randint(0, 99999))
        elif r < 0.45:
            c = "%.2f" % (rng.random() * 1000)
        elif r < 0.55:
            c = date_time(rng)
        elif r < 0.62:
            c = email(rng)
        elif r < 0.7:
            c = rng.choice(("active", "pending", "closed", "shipped", "yes", "no", "N/A", ""))
        elif r < 0.78:
            c = script_text(rng, rng.choice(SCRIPTS))[:30]
        else:
            c = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 4)))
        if sep == "," and ("," in c or '"' in c or rng.random() < 0.1):
            c = '"' + c.replace('"', '""') + '"'
        cells.append(c.replace("\t", " ").replace("\n", " ").replace("\r", " "))
    return sep.join(cells)


def log_line(rng: random.Random) -> str:
    ts = "%04d-%02d-%02d %02d:%02d:%02d" % (rng.randint(2020, 2026), rng.randint(1, 12), rng.randint(1, 28),
                                            rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
    lvl = rng.choice(("INFO", "DEBUG", "WARN", "WARNING", "ERROR", "TRACE", "info", "error"))
    mod = _nm(rng)
    r = rng.randint(0, 7)
    if r == 0:
        return '%s - - [%02d/%s/%04d:%02d:%02d:%02d +0000] "%s %s HTTP/1.1" %d %d "%s" "%s"' % (
            ipv4(rng), rng.randint(1, 28), _MONTHS[rng.randint(0, 11)][:3], rng.randint(2020, 2026), rng.randint(0, 23),
            rng.randint(0, 59), rng.randint(0, 59), rng.choice(("GET", "GET", "POST", "PUT", "HEAD", "DELETE")),
            url_path(rng) + ("?" + query_string(rng) if rng.random() < 0.4 else ""),
            rng.choice((200, 200, 200, 204, 301, 302, 304, 400, 401, 403, 404, 500, 502)), rng.randint(0, 200000),
            rng.choice(("-", url(rng))), user_agent(rng))
    if r == 1:
        return "%s,%03d %s [%s] %s" % (ts, rng.randint(0, 999), lvl, mod, english_sentence(rng))
    if r == 2:
        return "%s %d %02d:%02d:%02d %s %s[%d]: %s" % (_MONTHS[rng.randint(0, 11)][:3], rng.randint(1, 28), rng.randint(0, 23),
                                                 rng.randint(0, 59), rng.randint(0, 59), _nm(rng), mod, rng.randint(100, 65000),
                                                 english_sentence(rng))
    if r == 3:
        return json.dumps({"ts": ts, "level": lvl.lower(), "msg": english_sentence(rng), "req_id": rand_hex(rng, 12),
                           "latency_ms": rng.randint(1, 3000)}, ensure_ascii=rng.random() < 0.5)
    if r == 4:
        return "%sT%s.%03dZ %5s %d --- [%s] %s.%s.%s : %s" % (ts[:10], ts[11:], rng.randint(0, 999), lvl, rng.randint(1, 99999),
                                                              rng.choice(("main", "http-nio-8080-exec-1", "pool-1-thread-3")),
                                                              rng.choice(("c", "o", "com")), _nm(rng), _camel(rng).capitalize(),
                                                              english_sentence(rng))
    if r == 5:
        return "I%02d%02d %s.%06d %d %s.go:%d] %s" % (rng.randint(1, 12), rng.randint(1, 28), ts[11:], rng.randint(0, 999999),
                                                      rng.randint(1, 99), mod, rng.randint(1, 999), english_sentence(rng))
    if r == 6:
        return "level=%s ts=%s caller=%s.go:%d msg=\"%s\" duration=%dms" % (lvl.lower(), ts.replace(" ", "T"), mod, rng.randint(1, 900),
                                                                              english_sentence(rng).replace('"', "'"), rng.randint(1, 5000))
    return "[%s] %s %s %s %d %.3fs" % (ts, lvl, rng.choice(("GET", "POST")), url_path(rng), rng.choice((200, 404, 500)), rng.random() * 3)


def stack_line(rng: random.Random) -> str:
    r = rng.randint(0, 7)
    if r == 0:
        return '  File "/app/%s/%s.py", line %d, in %s' % (_nm(rng), _nm(rng), rng.randint(1, 2000), _nm(rng))
    if r == 1:
        return "Traceback (most recent call last):"
    if r == 2:
        return "%s: %s" % (rng.choice(("ValueError", "KeyError", "TypeError", "RuntimeError", "IndexError")),
                           rng.choice(("invalid literal for int() with base 10: '%s'" % _nm(rng), "'%s'" % _nm(rng),
                                       "list index out of range", "unsupported operand type(s) for +: 'int' and 'str'")))
    if r == 3:
        return "\tat com.%s.%s.%s(%s.java:%d)" % (_nm(rng), _nm(rng), _camel(rng), _nm(rng).capitalize(), rng.randint(1, 900))
    if r == 4:
        return "    at %s.%s (/app/src/%s.js:%d:%d)" % (_nm(rng).capitalize(), _camel(rng), _nm(rng), rng.randint(1, 900), rng.randint(1, 80))
    if r == 5:
        return "goroutine %d [running]:" % rng.randint(1, 500)
    if r == 6:
        return "thread 'main' panicked at src/%s.rs:%d:%d:" % (_nm(rng), rng.randint(1, 900), rng.randint(1, 80))
    return "Exception in thread \"main\" java.lang.%s: %s" % (rng.choice(("NullPointerException", "IllegalStateException",
                                                                       "IllegalArgumentException")), english_sentence(rng))


def markdown_text(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.25:
        return "# %s\n\n%s" % (" ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 5))).capitalize(), english_sentence(rng))
    if r < 0.45:
        return "\n".join("- " + english_sentence(rng) for _ in range(rng.randint(2, 4)))
    if r < 0.6:
        return "See [%s](%s) for details." % (rng.choice(NOUNS), url(rng))
    if r < 0.75:
        return "Use `%s()` to %s the %s, then call `%s.%s()`." % (_camel(rng), rng.choice(VERBS), rng.choice(NOUNS), _nm(rng), _camel(rng))
    if r < 0.88:
        return "**%s:** %s *%s*" % (rng.choice(("Note", "Tip", "Warning", "Update")), english_sentence(rng), rng.choice(ADJS))
    return "| %s | %s |\n|---|---|\n| %d | %s |" % (rng.choice(NOUNS), rng.choice(NOUNS), rng.randint(1, 99), rng.choice(ADJS))


def html_with_entities(rng: random.Random) -> str:
    ent = ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;", "&nbsp;", "&copy;", "&mdash;", "&hellip;", "&euro;", "&rsquo;", "&reg;")
    s = english_sentence(rng)
    words = s.split(" ")
    for _ in range(rng.randint(1, 3)):
        e = rng.choice(ent)
        pos = rng.randint(0, len(words))
        if e == "&amp;":
            words.insert(pos, "&amp;")
        elif e in ("&#39;", "&rsquo;"):
            words.insert(pos, rng.choice(("it", "don", "can", "that")) + e + rng.choice(("s", "t", "ll")))
        elif e == "&lt;":
            words.insert(pos, "&lt;%d" % rng.randint(1, 9))
        else:
            words.insert(pos, e)
    s = " ".join(words)
    if rng.random() < 0.3:
        tag = rng.choice(("p", "b", "i", "em", "strong", "span", "li"))
        s = "<%s>%s</%s>" % (tag, s, tag)
    return s


def json_doc(rng: random.Random) -> str:
    def val(depth):
        r = rng.random()
        if depth > 0 and r < 0.2:
            return obj(depth - 1)
        if depth > 0 and r < 0.28:
            return [val(depth - 1) for _ in range(rng.randint(0, 4))]
        if r < 0.4:
            return rng.randint(-1000, 100000)
        if r < 0.5:
            return rng.choice((True, False, None))
        if r < 0.58:
            return round(rng.random() * 1000, rng.randint(1, 4))
        if r < 0.72:
            return script_text(rng)
        if r < 0.82:
            return english_sentence(rng)
        return rng.choice((email(rng), date_time(rng), url(rng), name(rng), rng.choice(IDENT_WORDS)))

    def obj(depth):
        return {rng.choice(JSON_KEYS): val(depth) for _ in range(rng.randint(1, 5))}

    doc = obj(rng.choice((0, 0, 1, 1, 2))) if rng.random() < 0.85 else [val(1) for _ in range(rng.randint(1, 4))]
    return json.dumps(doc, ensure_ascii=rng.random() < 0.5,
                      separators=rng.choice(((",", ":"), (", ", ": "))))


def id_token(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.25:
        return "%s-%s-%s-%s-%s" % (rand_hex(rng, 8), rand_hex(rng, 4), rand_hex(rng, 4), rand_hex(rng, 4), rand_hex(rng, 12))
    if r < 0.45:
        return rand_hex(rng, rng.choice((32, 40, 64)))
    if r < 0.65:
        def seg(d):
            return base64.urlsafe_b64encode(json.dumps(d, separators=(",", ":")).encode()).decode().rstrip("=")
        return "%s.%s.%s" % (seg({"alg": rng.choice(("HS256", "RS256", "ES256")), "typ": "JWT"}),
                             seg({"sub": rand_hex(rng, 8), "name": name(rng), "iat": rng.randint(1500000000, 1900000000)}),
                             rand_b64(rng, 43, url=True))
    if r < 0.85:
        s = base64.b64encode(bytes(rng.randint(0, 255) for _ in range(rng.randint(6, 90)))).decode()
        return s if rng.random() < 0.6 else s.replace("+", "-").replace("/", "_").rstrip("=")
    return rng.choice(("sk_test_", "pk_live_", "ghp_", "AKIA", "")) + rand_alnum(rng, rng.randint(16, 36))


def phone(rng: random.Random) -> str:
    return rng.choice((
        "+1 (%03d) %03d-%04d" % (rng.randint(200, 999), rng.randint(200, 999), rng.randint(0, 9999)),
        "+44 20 %04d %04d" % (rng.randint(1000, 9999), rng.randint(1000, 9999)),
        "%03d-%03d-%04d" % (rng.randint(200, 999), rng.randint(200, 999), rng.randint(0, 9999)),
        "+82 10-%04d-%04d" % (rng.randint(0, 9999), rng.randint(0, 9999)),
        "+81 90-%04d-%04d" % (rng.randint(0, 9999), rng.randint(0, 9999)),
        "+49 30 %d" % rng.randint(1000000, 99999999),
        "0%d %d %d" % (rng.randint(1, 9), rng.randint(100, 999), rng.randint(1000, 9999)),
    ))


def address(rng: random.Random) -> str:
    return rng.choice((
        "%d %s %s, %s, %s %05d" % (rng.randint(1, 9999), rng.choice(IDENT_WORDS).capitalize(),
                                   rng.choice(("St", "Ave", "Rd", "Blvd", "Lane", "Dr", "Way")), brand(rng).capitalize(),
                                   rng.choice(("CA", "NY", "TX", "WA", "IL", "MA")), rng.randint(10000, 99999)),
        "%s %d, %05d %s" % (rng.choice(("Hauptstraße", "Bahnhofstraße", "Gartenweg", "Rue de la Paix", "Calle Mayor")),
                            rng.randint(1, 200), rng.randint(10000, 99999), rng.choice(("Berlin", "München", "Paris", "Madrid"))),
        "Apt %d, %d %s Street" % (rng.randint(1, 40), rng.randint(1, 999), rng.choice(IDENT_WORDS).capitalize()),
        "%s %d-%d" % (script_text(rng, rng.choice(("ja", "ko", "zh")))[:12], rng.randint(1, 9), rng.randint(1, 30)),
    ))


# ==========================================================================
# Percent-encoding helpers for benign values (single layer only)
# ==========================================================================

_UNRESERVED = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


def pct_encode_text(s: str, space_as_plus: bool = True, upper: bool = True) -> str:
    fmt = "%%%02X" if upper else "%%%02x"
    out = []
    for c in s.encode("utf-8"):
        if c in _UNRESERVED:
            out.append(chr(c))
        elif space_as_plus and c == 0x20:
            out.append("+")
        else:
            out.append(fmt % c)
    return "".join(out)


def encoded_value(rng: random.Random) -> str:
    """`key=` plus a single-layer percent-encoded text value (many escapes)."""
    src = rng.choice((script_text, english_sentence, name, email, address, title_line, mixed_text))(rng)
    src = src[:rng.randint(8, 100)].strip() or "hello world"
    return rng.choice(PARAM_NAMES) + "=" + pct_encode_text(src, rng.random() < 0.6, rng.random() < 0.8)


def pair_source_value(rng: random.Random) -> str:
    """A benign value that needs encoding (spaces, punctuation or non-ASCII)."""
    r = rng.random()
    if r < 0.3:
        return script_text(rng)[:60]
    if r < 0.5:
        s = english_sentence(rng)
        return s[:rng.randint(8, 80)].strip() or "hello world"
    if r < 0.6:
        return email(rng)
    if r < 0.7:
        return name(rng)
    if r < 0.8:
        return "%s %s" % (money(rng), percent(rng))
    if r < 0.9:
        return url(rng)
    return title_line(rng)[:60]


# ==========================================================================
# Benign value pool for carriers, and standalone benign samples
# ==========================================================================

def benign_value(rng: random.Random) -> str:
    """A benign value to embed in the shared carriers."""
    r = rng.random()
    if r < 0.12:
        return query_value(rng)
    if r < 0.22:
        return english_sentence(rng)
    if r < 0.32:
        return script_text(rng)
    if r < 0.38:
        return name(rng)
    if r < 0.43:
        return email(rng)
    if r < 0.47:
        return phone(rng)
    if r < 0.52:
        return date_time(rng)
    if r < 0.57:
        return id_token(rng)
    if r < 0.62:
        return rng.choice((ipv4(rng), ipv4(rng), ipv6(rng), "localhost", "127.0.0.1", neutral_host(rng)))
    if r < 0.66:
        return url(rng)
    if r < 0.7:
        return rng.choice((posix_path(rng), windows_path(rng), filename(rng)))
    if r < 0.74:
        return " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 5)))
    if r < 0.78:
        return money(rng) + rng.choice(("", " " + percent(rng) + " off!", " (" + percent(rng) + ")"))
    if r < 0.82:
        return title_line(rng)
    if r < 0.85:
        return address(rng)
    if r < 0.88:
        return version_constraint(rng)
    if r < 0.91:
        return mixed_text(rng)
    if r < 0.94:
        return "#" + rand_hex(rng, rng.choice((3, 6)))
    if r < 0.97:
        return emoji(rng) + " " + english_sentence(rng)
    return rng.choice(("%s & %s" % (rng.choice(NOUNS), rng.choice(NOUNS)), "%s; %s; %s" % tuple(rng.choice(NOUNS) for _ in range(3)),
                       "%s | %s | %s" % tuple(rng.choice(NOUNS) for _ in range(3))))


FAMILIES: List[Tuple[str, Callable[[random.Random], str], int]] = [
    ("english", lambda r: english_text(r, 3), 12),
    ("script", lambda r: script_text(r), 12),
    ("mixed", lambda r: mixed_text(r), 5),
    ("title", title_line, 2),
    ("code", code_line, 6),
    ("arith", arithmetic, 3),
    ("config", config_line, 3),
    ("regex", regex, 2),
    ("version", version_constraint, 2),
    ("header", header_value, 7),
    ("mime", mime_type, 1),
    ("posix_path", posix_path, 2),
    ("windows_path", windows_path, 2),
    ("url", url, 4),
    ("url_path", lambda r: url_path(r, r.randint(1, 5)) + ("?" + query_string(r) if r.random() < 0.5 else ""), 4),
    ("query", query_string, 5),
    ("form", form_body, 4),
    ("encoded", encoded_value, 4),
    ("cookie", cookie_header, 2),
    ("csv", lambda r: csv_row(r, ","), 2),
    ("tsv", lambda r: csv_row(r, "\t"), 1),
    ("log", log_line, 3),
    ("stack", stack_line, 2),
    ("date", date_time, 2),
    ("id", id_token, 2),
    ("json", json_doc, 5),
    ("html_entities", html_with_entities, 2),
    ("markdown", markdown_text, 2),
    ("phone", phone, 1),
    ("address", address, 1),
    ("email", email, 1),
    ("emoji", lambda r: " ".join(emoji(r) for _ in range(r.randint(1, 4))) + (" " + english_sentence(r) if r.random() < 0.5 else ""), 1),
]
_FAM_NAMES = [f[0] for f in FAMILIES]
_FAM_FUNCS = [f[1] for f in FAMILIES]
_FAM_WEIGHTS = [f[2] for f in FAMILIES]


def sample(rng: random.Random) -> Tuple[bytes, str]:
    """One standalone benign sample: (bytes, family name). Not yet checked."""
    i = rng.choices(range(len(FAMILIES)), _FAM_WEIGHTS)[0]
    return _FAM_FUNCS[i](rng).encode("utf-8"), _FAM_NAMES[i]


def corruption_base(rng: random.Random) -> bytes:
    """A benign sample biased toward non-ASCII and percent-encoded content."""
    r = rng.random()
    if r < 0.3:
        return script_text(rng).encode("utf-8")
    if r < 0.6:
        return rng.choice((encoded_value, form_body, lambda x: url_path(x, 2) + "?" + query_string(x)))(rng).encode("utf-8")
    return sample(rng)[0]


# ==========================================================================
# Malformed rule and corruption
# ==========================================================================

_HEXB = frozenset(b"0123456789abcdefABCDEF")


def _valid_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def has_bad_control(b: bytes) -> bool:
    return any(c < 0x20 and c not in (0x09, 0x0A, 0x0D) for c in b)


def invalid_pct_at(b: bytes, j: int) -> bool:
    return b[j] == 0x25 and not (j + 2 < len(b) and b[j + 1] in _HEXB and b[j + 2] in _HEXB)


def has_invalid_pct_escape(b: bytes) -> bool:
    return any(invalid_pct_at(b, j) for j in range(len(b)) if b[j] == 0x25)


def is_malformed(b: bytes) -> bool:
    """The malformed rule: invalid UTF-8, a bad C0 control, or an invalid %-escape."""
    return (not _valid_utf8(b)) or has_bad_control(b) or has_invalid_pct_escape(b)


def _boundaries(b: bytes) -> List[int]:
    return [i for i in range(len(b) + 1) if i == len(b) or not (0x80 <= b[i] <= 0xBF)]


def _insert(rng: random.Random, b: bytes, junk: bytes) -> bytes:
    pos = rng.choice(_boundaries(b))
    return b[:pos] + junk + b[pos:]


CORRUPTIONS = ("truncate_mb", "stray_cont", "bad_lead", "bad_second", "nul", "c0", "bad_pct", "binary",
               "legacy_char", "truncate_end", "ansi", "bad_pct_end", "utf16_frag")
_LEGACY = ("cp1252", "latin-1", "cp1251", "iso8859_7", "euc_kr", "shift_jis", "gbk", "cp1256", "cp1255", "tis_620")


def _apply(rng: random.Random, clean: bytes, kind: str) -> Optional[bytes]:
    if kind == "truncate_mb":
        leads = [i for i, c in enumerate(clean) if c >= 0xC2]
        if not leads:
            return None
        i = rng.choice(leads)
        c = clean[i]
        ln = 2 if c < 0xE0 else 3 if c < 0xF0 else 4
        keep = rng.randint(1, ln - 1)
        return clean[:i + keep] + clean[i + ln:]
    if kind == "stray_cont":
        return _insert(rng, clean, bytes(rng.randint(0x80, 0xBF) for _ in range(rng.randint(1, 3))))
    if kind == "bad_lead":
        junk = bytes((rng.choice((0xC0, 0xC1, 0xF5, 0xF6, 0xF7, 0xF8, 0xFB, 0xFE, 0xFF)),))
        if rng.random() < 0.5:
            junk += bytes((rng.randint(0x80, 0xBF),))
        return _insert(rng, clean, junk)
    if kind == "bad_second":
        r = rng.random()
        if r < 0.5:
            # lead byte directly followed by an ASCII byte
            cands = [i for i in range(len(clean)) if clean[i] < 0x80 and (i == 0 or clean[i - 1] < 0x80 or clean[i - 1] >= 0xC0)]
            if not cands:
                return None
            i = rng.choice(cands)
            return clean[:i] + bytes((rng.choice((0xC3, 0xD0, 0xE2, 0xE3, 0xE4, 0xF0)),)) + clean[i:]
        if r < 0.75:
            return _insert(rng, clean, b"\xed" + bytes((rng.randint(0xA0, 0xBF), rng.randint(0x80, 0xBF))))  # surrogate
        return _insert(rng, clean, b"\xf4" + bytes((rng.randint(0x90, 0xBF), rng.randint(0x80, 0xBF), rng.randint(0x80, 0xBF))))
    if kind == "nul":
        if rng.random() < 0.3 and b" " in clean:
            idx = [i for i, c in enumerate(clean) if c == 0x20]
            i = rng.choice(idx)
            return clean[:i] + b"\x00" + clean[i + 1:]
        return _insert(rng, clean, b"\x00")
    if kind == "c0":
        ctl = rng.choice((0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0B, 0x0C, 0x0E, 0x10, 0x15, 0x1A, 0x1B, 0x1C, 0x1F))
        return _insert(rng, clean, bytes((ctl,)))
    if kind == "bad_pct":
        esc = [j for j in range(len(clean) - 2) if clean[j] == 0x25 and clean[j + 1] in _HEXB and clean[j + 2] in _HEXB]
        if not esc:
            return None
        j = rng.choice(esc)
        nonhex = rng.choice(b"ghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ")
        mode = rng.choice(("truncate", "letters", "letters", "drop", "first"))
        if mode == "truncate":
            out = clean[:j + 2] + clean[j + 3:]          # %4 + rest
        elif mode == "letters":
            out = clean[:j + 1] + bytes((nonhex, rng.choice(b"0123456789abcdefghxyz"))) + clean[j + 3:]
        elif mode == "first":
            out = clean[:j + 1] + bytes((clean[j + 1], nonhex)) + clean[j + 3:]
        else:
            out = clean[:j + 1] + clean[j + 3:]          # bare %
        return out if invalid_pct_at(out, j) else None
    if kind == "legacy_char":
        # one character re-encoded in a legacy code page, as a mixed-encoding join does
        text = clean.decode("utf-8")
        idx = [i for i, ch in enumerate(text) if ord(ch) > 0x7F]
        rng.shuffle(idx)
        for i in idx[:8]:
            for codec in rng.sample(_LEGACY, len(_LEGACY)):
                try:
                    legacy = text[i].encode(codec)
                except UnicodeEncodeError:
                    continue
                out = text[:i].encode("utf-8") + legacy + text[i + 1:].encode("utf-8")
                if not _valid_utf8(out):
                    return out
        return None
    if kind == "truncate_end":
        # the field was cut in the middle of its last multi-byte character
        leads = [i for i, c in enumerate(clean) if c >= 0xC2]
        if not leads:
            return None
        i = leads[-1]
        c = clean[i]
        ln = 2 if c < 0xE0 else 3 if c < 0xF0 else 4
        if len(clean) - (i + ln) > 2:
            return None
        return clean[:i + rng.randint(1, ln - 1)]
    if kind == "ansi":
        seq = rng.choice((b"\x1b[31m", b"\x1b[0m", b"\x1b[1;33m", b"\x1b[2K", b"\x1b[H", b"\x1b[?25h", b"\x1b]0;x\x07"))
        return _insert(rng, clean, seq)
    if kind == "bad_pct_end":
        # a percent-escape cut at the end of the field
        return clean + rng.choice((b"%", b"%2", b"%E", b"%C", b"%e2%8"))
    if kind == "utf16_frag":
        word = rng.choice(("id", "ok", "name", "x", "file", "data"))
        return _insert(rng, clean, word.encode(rng.choice(("utf-16-le", "utf-16-be"))))
    # binary splice
    run = bytes(rng.randint(0, 255) for _ in range(rng.randint(3, 16)))
    return _insert(rng, clean, run)


def corrupt(rng: random.Random, clean: bytes, kind: Optional[str] = None) -> bytes:
    """Apply exactly one structural damage to `clean`; result satisfies is_malformed()."""
    has_mb = any(c >= 0xC2 for c in clean)
    has_pct = any(clean[j] == 0x25 and clean[j + 1] in _HEXB and clean[j + 2] in _HEXB for j in range(len(clean) - 2))
    for attempt in range(20):
        k = kind if (kind is not None and attempt < 5) else None
        if k is None:
            weights = [3 if has_mb else 0, 2, 2, 2, 2, 2, 7 if has_pct else 0, 1,
                       2 if has_mb else 0, 1 if has_mb else 0, 1, 1, 1]
            k = rng.choices(CORRUPTIONS, weights)[0]
        out = _apply(rng, clean, k)
        if out is not None and out != clean and is_malformed(out):
            return out
    return _insert(rng, clean, b"\x00")


# ==========================================================================
# Final benign check
# ==========================================================================

_FORBIDDEN = re.compile(
    r"\.\.[/\\]|%2e%2e|\{\{|\$\{|#\{|<%|\{%|<\s*/?\s*(script|iframe|svg|img|object|embed|body|math|meta|link|style|base)\b"
    r"|javascript\s*:|vbscript\s*:|data\s*:\s*text/html|\bon[a-z]{3,15}\s*=|<!\s*(doctype|entity)|<\?xml"
    r"|/etc/(passwd|shadow|hosts|group)|win\.ini|boot\.ini|system32|file://|/proc/self|%00"
    r"|%25[0-9a-f]{2}|%u[0-9a-f]{4}|(&#x?[0-9a-f]+;){3}|(\\x[0-9a-f]{2}){3}|\\u00[0-7][0-9a-f]|(\\[0-7]{3}){3}"
    r"|\bunion\b[\s\S]{0,20}\bselect\b|'\s*(or|and)\s+(\d|['\"]?\w*['\"]?\s*(=|<|>|\blike\b))|\b(sleep|benchmark|pg_sleep|extractvalue|updatexml|load_file)\s*\("
    r"|waitfor\s+delay|information_schema|;\s*(drop|truncate|alter)\s+table\b|;\s*(insert\s+into|delete\s+from|shutdown)\b"
    r"|[a-z0-9]/\*[^*]{0,12}\*/[a-z0-9]|--\s*$"
    r"|[;&|`]\s*(cat|id|whoami|uname|wget|curl|nc|ncat|bash|sh|zsh|ping|powershell|cmd|ls|rm|python|perl|nslookup|certutil)"
    r"(\s+(-\w|/|\$|[\w.-]+\.(com|net|org|io|ru|cn|exe)\b|\d+\.\d+)|\s*$|\s*[;&|`)])"
    r"|\$\(|/dev/tcp|\bifs\b|\bjndi\b|<#|#set\(|\bT\(java|java\.lang\.runtime|__class__|__globals__|getruntime"
    r"|(\r|\n|%0d|%0a)\s*(set-cookie|location|content-length|transfer-encoding|bcc|cc|to|from|refresh|x-[a-z-]+|host|content-type|subject)\s*:"
    r"|(\r|\n|%0d|%0a)\s*(get|post|put|head)\s+/",
    re.IGNORECASE,
)
# Forms that the attack grammars in `attacks.py` generate beyond the textbook
# ones above. A benign sample must never match one of them.
_FORBIDDEN_EXTRA = re.compile(
    # SQL: quote or parenthesis breakouts, dialect functions, stacked statements
    r"(['\")`]|^)(/\*[^*]*\*/|%20|%09|%0a|\+|\s)*(or|and|xor|\|\||&&)(/\*[^*]*\*/|%20|%09|%0a|\+|\s)+"
    r"(\d|['\"(]|not\b|true\b|false\b|null\b|[a-z_]+\s*\()"
    r"|['\")`](\s|%0[9a]|%20|\+|/\*[^*]*\*/)*;(\s|%0[9a]|%20|\+|/\*[^*]*\*/)*(declare|update|exec|insert|delete|select|drop|create|grant|copy|truncate|shutdown"
    r"|sleep|benchmark|waitfor|dbms_\w+)\b|^\d+(\s|%0[9a]|%20|\+|/\*[^*]*\*/)*;(\s|%0[9a]|%20|\+|/\*[^*]*\*/)*(sleep|benchmark|waitfor|shutdown|declare|drop)\b"
    r"|['\")`]\s*(%23|--(\+|%20|$| //| -))|(['\")`]|^)(\s|%0[9a]|%20|\+|/\*[^*]*\*/)*(union|having)(\s|%0[9a]|%20|\+|/\*[^*]*\*/)+(all|select|\d|')"
    r"|(%0[9a]|/\*[^*]*\*/)(union|select|from|where|and|or|xor|having|group|order|insert|update|delete|drop|limit)\b"
    r"|\b(union|select|from|where|and|or|xor|having|insert|into|update|delete|drop|table|all|by|set)(%0[9a]|/\*[^*]*\*/)"
    r"|(%20|\+)(union(%20|\+)(all(%20|\+))?select|having(%20|\+)\d|(sleep|benchmark)\(|waitfor(%20|\+)delay)"
    r"|\b(dbms_\w+|utl_http|utl_inaddr|gtid_subset|xp_cmdshell|xp_dirtree|sp_configure|randomblob|ctxsys|xmltype"
    r"|group_concat|string_agg|sqlite_master|pg_catalog|pg_shadow|sysobjects|@@version|@@datadir)\b"
    r"|\b(order|group)(\s|%0a|%20|%09|\+|/\*\*/)+by(\s|%0a|%20|%09|\+|/\*\*/)+\d+|\binto\s+(outfile|dumpfile)\b|#\s*$|(?<!\*)/\*\s*$|-- [a-z/+-]?$"
    # markup: event handlers after any separator, sinks, and the less common tags
    r"|(?<![a-z])on[a-z]{3,20}\s*=|alert\s*[(`]|eval\s*\(|atob\(|\[['`]al['`]|prompt\s*\(|confirm\s*\(|document\.(cookie|domain|write|location)"
    r"|<\s*/?\s*(marquee|form|applet|xmp|noscript|template|isindex|frameset|keygen|details|video|audio|input|textarea"
    r"|button|select|source|track|table|xss|animate|set|maction|iframe|frame)\b|expression\s*\(|xi:include"
    r"|formaction|codebase\s*=|srcdoc\s*=|http-equiv|<!\[cdata\[|\]\]>|-->|</(title|textarea|noscript|option)>"
    # path traversal: encoded separators, overlong and wide escapes, stream wrappers, sensitive files
    r"|\.\.(%2f|%5c|;/|\\/|%252f|%255c|%c0|%c1|%e0|%u2215|%u2216|%%)|%c0%a[ef]|%c0%2e|%c[01]%[89ab][0-9a-f]|%e0%80%[89ab][0-9a-f]|%uff0e|\uff0e\uff0e"
    r"|(php|phar|zip|compress\.zlib|expect|netdoc)://|jar:(file|https?):|\\\\\?\\|[\\/](c|d|admin|ipc)\$|%2e"
    r"|/etc/(sudoers|crontab|issue|hostname|fstab|resolv\.conf|ssh/|mysql/|nginx/|apache2/|httpd/|php\.ini)|/proc/"
    r"|(^|[\s=:'\x22])/root/\.|\.ssh/(id_|authorized)|\.aws/credentials|serviceaccount|web\.config|wp-config|\.htpasswd|\.git/(config|head)"
    r"|web-inf|::\$data"
    # command injection: separators followed by a command, shell substitutions
    r"|([;&|`]|%0a|%0d)\s*(ls|id|whoami|uname|wget|curl|nc|sleep|timeout|ver|kill|python3?|php|ruby|awk|busybox|tftp|telnet|socat|mkfifo|chmod|touch|echo|dir|type"
    r"|net|ipconfig|systeminfo|tasklist|reg|del|wmic|mshta|rundll32|regsvr32|bitsadmin|hostname|env|printenv|ps|ifconfig"
    r"|netstat|dig|find|crontab|sudo|tail|head|xxd|cp|sh|bash|pwd|set)\b(?![=;.(-])|\$\{?ifs|base64\s+-d|/bin/(sh|bash)\b|/bin/ca[t\[]"
    r"|(\r|\n)\s*(whoami|net user|sudo -l|ps aux|ps -ef|ss -tlnp|netstat -an|nslookup|ipconfig|ifconfig|uname -a"
    r"|cmd\.exe|powershell|ip a|cat /|hostname|python3? -c)\b|cmd\.exe\s*/c|([;&|`]|%0a|%0d)\s*(cmd|powershell|nslookup|last|ss|ip|rm|cat|ping|uname)\b(?![=;.(-])"
    r"|\w`\s*(id|whoami|uname|ls|cat|ip|last|hostname|env)\b|\$\(\s*(id|whoami|uname|cat|ls|curl|wget)\b|[;&|]\s*\{\s*(cat|whoami|id|curl|wget|busybox|ls|uname|env|nc|bash|sh)\b"
    r"|[<>]\((?!\))|\w\^\w+\^|\?\?\?/|\\(at|oami)|\$'\\\d|\{[a-z]+,-"
    # template injection: delimiters and engine idioms
    r"|\[\[\$|\[\[\s*'?\d+'?\s*[*+-]|\[\(\s*[a-z]|\[=|\*\{|@\(|@\{|<#=|\{\$|\{php\}|smarty|#foreach|xwork|_memberaccess|ognl|getclass"
    r"|forname|\.constructor\s*\(|\bconstructor\.constructor|__import__|popen|process\.start"
    r"|\$\{?(env|sys|java|ctx|main|lower|upper|date):|\{=?\s*\d+\s*[*+-]\s*'?\d+'?\s*\}|\{math\b|\{if\s|\{system\("
    # header injection: a raw, encoded, or escaped line break followed by a header or a request line
    r"|(\r|\n|%0d|%0a|\\r|\\n|%c4%8a|%e5%98%8a|%e5%98%8d|%u000a)(%20|%09|\s)*(set-cookie|location|content-[a-z-]+|transfer-encoding"
    r"|x-[a-z-]+|host|refresh|access-control-[a-z-]+|cache-control|link|authorization|connection|upgrade|cookie|reply-to"
    r"|mime-version|bcc|cc|to|from|subject|vary|info|warn|error)\s*:"
    r"|(\r|\n|%0d|%0a)(%20|\s)*\d+\.\d+\.\d+\.\d+ - - \[|(%0d|%0a|\r|\n)(%20|\s)*info user|http/1\.[01] \d{3}"
    # obfuscating encodings from `obfuscation.py`
    r"|(\\[0-9a-f]{2,6}\s?){3}|\\u\{[0-9a-f]+\}|\+A[CDEFG][0-9A-Za-z+/]{0,6}-|/\*!"
    r"|[a-z][\u200b-\u200d\u2060\ufeff\u00ad\u034f\u180e][a-z]|[\U0001d400-\U0001d7ff\ufe50-\ufe6b]"
    r"|[\u2039\u203a\u2032\u2033\u2044\u2215\u2216\u29f8\u276e\u276f\u02c2\u02c3\u02bc\u02ba\u037e\u2024]"
    r"|[a-z][\u0300-\u036f]|(&#x?[0-9a-f]+){3}|['\"`]\s*\+\s*['\"`][a-z]",
    re.IGNORECASE,
)
_FULLWIDTH_BAD = re.compile("[\uff02-\uff07\uff0a\uff0b\uff0d-\uff19\uff1c-\uff1e\uff20-\uff5d]")
_MIXED_SCRIPT = re.compile("[A-Za-z][\u0400-\u04ff\u0370-\u03ff]|[\u0400-\u04ff\u0370-\u03ff][A-Za-z]")


@functools.lru_cache(maxsize=1)
def _sensitive_reference() -> re.Pattern:
    """A reference to a file that the attack grammars treat as a traversal target.

    The attack grammars label such a reference `path_traversal`, so benign text
    never contains one. Matched after backslashes become slashes.
    """
    import attacks  # imported here because `attacks` imports this module

    names = {name.lower().replace("\\", "/") for name in attacks.SENSITIVE_FILES + attacks.WINDOWS_FILES}
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(r"(?:^|[/=:]|%2f)(?:" + alternatives + r")(?![\w-])", re.IGNORECASE)


def benign_ok(b: bytes) -> bool:
    """Final check for every benign sample."""
    if not (1 <= len(b) <= 512):
        return False
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if has_bad_control(b):
        return False
    if _sensitive_reference().search(s.replace("\\", "/")):
        return False
    if _FORBIDDEN.search(s) or _FORBIDDEN_EXTRA.search(s) or _FULLWIDTH_BAD.search(s) or _MIXED_SCRIPT.search(s):
        return False
    return True
