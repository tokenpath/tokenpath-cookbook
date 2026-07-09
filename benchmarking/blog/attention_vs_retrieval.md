# Where attention beats retrieval: citations similarity can't find

A recurring pattern in the data: TokenPath cites the *right* sentence where
embedding retrieval cites a *similar-looking* one. All three examples below are
real (LongBench-Cite / hotpotqa, same Sonnet-5 answer, `gemini-2.5-flash` judge):
TokenPath's citation was judged **fully supported**, embedding's was judged **not
supported**. And they all fail retrieval for the same reason — **coreference**.

The fact you need lives in a sentence that refers to the entity by a pronoun or a
short-form name ("the mascot", "she", "Roth"). The sentence that's most
*lexically similar* to the answer is the entity's naming/definition sentence —
which doesn't contain the fact. Embedding retrieval is pulled to the name;
TokenPath's attention — the generating model's own focus — follows the
coreference to the sentence that actually states the fact.

---

### 1. The Oregon Duck's colors

**Q:** What colors are worn by the Oregon Duck?
**Answer statement:** *"The Oregon Duck mascot wears green and yellow colors."*

- ✅ **TokenPath →** *"**The mascot** wears a green and yellow costume, and a green and yellow beanie cap with the word 'Oregon' written on it."* — the sentence that actually states the colors.
- ❌ **Embedding →** *"**The Oregon Duck** is the mascot of the University of Oregon Ducks athletic program, based on Disney's Donald Duck character…"* — the naming sentence. Highest similarity to "Oregon Duck", zero mention of colors.

*Retrieval matched the name "Oregon Duck"; the color fact is in a sentence that calls it "the mascot", so retrieval never surfaces it. Attention resolved the coreference.*

---

### 2. Where Henry Roth was born

**Q:** Henry Roth and Lewis Mumford are both this nationality?
**Answer statement:** *"Henry Roth was born in Austro-Hungary (present-day Ukraine) and immigrated to the U.S."*

- ✅ **TokenPath →** *"**Roth** was born in Tysmenitz near Stanislawow, Galicia, Austro-Hungary (now known as Tysmenytsia, near Ivano-Frankivsk, Galicia, Ukraine)."* — the birthplace sentence.
- ❌ **Embedding →** *"**Henry Roth** (February 8, 1906 – October 13, 1995) was an American novelist and short story writer…"* — the biography intro. Matched the full name; says nothing about where he was born.

*The birthplace lives in a sentence keyed on the surname "Roth" and full of place names ("Tysmenitz", "Galicia") absent from the answer. Retrieval locked onto the full-name intro; attention found the fact.*

---

### 3. Rønnaug Alten's stage debut

**Q:** …the play in which Rønnaug Alten made her stage debut at Den Nationale Scene in 1930 as "Viola"?
**Answer statement:** *"Rønnaug Alten made her stage debut at Den Nationale Scene in 1930 as 'Viola' in Shakespeare's play Twelfth Night."*

- ✅ **TokenPath →** *"…**She** made her stage debut at Den Nationale Scene in 1930 as 'Viola' in Shakespeare's play Twelfth Night."* — the exact fact (Twelfth Night included).
- ❌ **Embedding →** *"**Rønnaug Alten** (9 February 1910 – 20 January 2001) was a Norwegian actress and stage instructor."* — the definition sentence. Matched the name; no debut, no play.

*The debut sentence starts with the pronoun "She", so it's lexically distant from "Rønnaug Alten". Retrieval took the name-bearing intro; attention followed the pronoun.*

---

## Why this matters

Embedding retrieval has no model of what the answer is *about* — only surface
similarity. So when a fact is stated about an entity referred to indirectly, the
retriever is systematically pulled toward the wrong (but similar-looking)
sentence. TokenPath's citation **is** the generating model's attention, which
already tracked the entity across the coreference chain — so it lands on the
sentence that actually carries the fact. That's a class of citation retrieval
structurally can't get right, and it shows up across the benchmark (256 such
statements where TokenPath was supported and embedding was not).

*(Raw records: `data/attention_vs_retrieval_examples.json`.)*
