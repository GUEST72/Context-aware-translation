Step	What happens
1. POS-tag	NLTK tags every body paragraph → keeps only NN/NNS/NNP/NNPS and JJ/JJR/JJS
2. Pre-process	Lingo filter on original token (acronyms, camelCase, dots, hyphens, known suffix fragments) → lemmatize + lowercase → min 4 chars
3. Common filter	NLTK stopwords ∪ Google-1000 words ∪ top-3000 Brown+Reuters words
4. NER filter	spaCy entity spans across entire book → blocklist
5. Fragment guard	If lemma not in NLTK English dictionary AND freq < 7 → reject (catches PDF hyphenation tails like rithm, ator)
6. Weirdness score	TF_book / TF_background — ranked descending, threshold ≥ 1.5

The architecture
OFFLINE (runs once after PDF upload)
─────────────────────────────────────────────────────
For each term T in terminology list:
  1. Find first 2 occurrences in different sections
  2. Collect all paragraphs from those sections → definition window
  3. Split window into sentences
  4. Embed each sentence → store as {term: [(sentence, embedding, metadata)]}
  Save to definition_space.json

ONLINE (at translation time, last layer before translation prompt)
─────────────────────────────────────────────────────
Given term T to translate:
  1. Load T's definition window from definition_space
  2. Embed ["What is T?", "T is", "T refers to"] → 3 query vectors
  3. Retrieve top-3 sentences per query → deduplicate → 5-7 candidates
  4. Rerank by: subject position + first-in-paragraph + soft IS-A verb + section proximity
  5. Take top-2 sentences → append to translation prompt as "Definition context"

No rigid pattern matching — pure semantic retrieval handles any definition style
Focused space — you're not searching the whole book, only the section where the term lives
Sentence-level precision — output is a clean 1-2 sentence definition, not a full paragraph
Integrates naturally — the definition space becomes a context enrichment layer for translation, exactly as you described


examples
Done in 23.4s
Single tokens   : 737
Multi-word phrases: 716
Total           : 1453

========================================================================
TOP 40 PHRASES  (sorted by frequency)
========================================================================
PHRASE                                 FREQ    PAGES
------------------------------------------------------------------------
client code                            85      54, 55, 56, 57  (+60)
factory method                         53      6, 72, 74, 75  (+18)
abstract factory                       34      87, 88, 89, 90  (+17)
same interface                         31      18, 75, 84, 115  (+25)
other objects                          27      15, 47, 90, 107  (+19)
common interface                       20      43, 76, 109, 111  (+16)
service object                         20      155, 161, 209, 219  (+8)
design patterns                        18      3, 7, 25, 26  (+8)
single responsibility principle        18      5, 50, 51, 54  (+14)
concrete classes                       18      40, 42, 88, 100  (+12)
template method                        18      87, 381, 382, 384  (+7)
same time                              17      18, 23, 47, 49  (+10)
other hand                             17      18, 25, 87, 113  (+13)
business logic                         16      66, 121, 142, 206  (+11)
 open closed principle                14      86, 102, 161, 176  (+10)
wrapped object                         14      152, 162, 190, 196  (+6)
base class                             13      61, 75, 79, 83  (+5)
extrinsic state                        13      223, 224, 225, 227  (+2)
existing code                          12      27, 31, 40, 44  (+8)
component interface                    11      183, 185, 188, 189  (+3)
mediator object                        11      268, 288, 305, 307  (+5)
memento class                          11      328, 329, 330, 331  (+2)
strategy pattern                       10      48, 54, 175, 358  (+3)
return type                            10      56, 75, 77, 85  (+4)
various ways                           10      108, 126, 133, 264  (+6)
director class                         10      109, 110, 111, 117  (+4)
object tree                            10      120, 181, 189, 190  (+4)
behavioral design pattern              10      251, 269, 290, 305  (+6)
mediator pattern                       10      307, 308, 311, 315  (+3)
one object                             9       21, 23, 46, 152  (+5)
most cases                             9       106, 145, 219, 224  (+5)
builder pattern                        9       107, 111, 114, 118  (+2)
singleton pattern                      9       138, 139, 143, 145
command pattern                        9       273, 275, 278, 284  (+1)
that class                             8       11, 96, 124, 132  (+4)
private fields                         8       59, 125, 134, 323  (+3)
ui elements                            8       95, 96, 307
new object                             8       124, 126, 130, 138  (+3)
that object                            8       125, 288, 335, 336  (+3)
composite pattern                      8       179, 180, 181, 183  (+3)

========================================================================
COMBINED TOP 40  (tokens + phrases by weirdness)
========================================================================
#    TERM                                   POS     FREQ
----------------------------------------------------------
1    subclass                               NOUN    127
2    client code                            PHRASE  85
3    flyweight                              NOUN    73
4    iterator                               NOUN    65
5    factory method                         PHRASE  53
6    originator                             NOUN    39
7    dialog                                 NOUN    36
8    abstract factory                       PHRASE  34
9    superclass                             NOUN    32
10   same interface                         PHRASE  31
11   iterators                              NOUN    30
12   other objects                          PHRASE  27
13   runtime                                NOUN    26
14   high-level                             ADJ     20
15   low-level                              ADJ     20
16   common interface                       PHRASE  20
17   service object                         PHRASE  20
18   functionality                          NOUN    19
19   initialization                         NOUN    18
20   design patterns                        PHRASE  18
21   single responsibility principle        PHRASE  18
22   concrete classes                       PHRASE  18
23   template method                        PHRASE  18
24   extrinsic                              ADJ     17
25   same time                              PHRASE  17
26   other hand                             PHRASE  17
27   business logic                         PHRASE  16
28    open closed principle                PHRASE  14
29   wrapped object                         PHRASE  14
30   base class                             PHRASE  13
31   extrinsic state                        PHRASE  13
32   backup                                 NOUN    12
33   existing code                          PHRASE  12
34   foreach                                NOUN    11
35   traversal                              NOUN    11
36   component interface                    PHRASE  11
37   mediator object                        PHRASE  11
38   memento class                          PHRASE  11
39   conditionals                           NOUN    10
40   filename                               NOUN    10

Root cause for the low-confidence group: their first occurrence is in a relations section ("works with X, Y, Z") rather than a dedicated introduction paragraph. Increasing window_paragraphs or adding a second occurrence window would pull in the actual definition. Want me to fix these three issues next?
the window size is so important need a reaserch  .