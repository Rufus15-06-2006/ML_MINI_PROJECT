"""MeSH keyword extraction with a word-level Trie (longest match, synonyms -> one concept)."""
import re
import xml.etree.ElementTree as ET
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
END = "$"

def tokenize(text):
    return TOKEN.findall(text.lower())

def load_mesh_synonyms(desc_xml):
    """Parse NLM descriptor file (descYYYY.xml) -> {entry term string: descriptor name}."""
    syn = {}
    for _, el in ET.iterparse(desc_xml, events=("end",)):
        if el.tag == "DescriptorRecord":
            name = el.findtext("DescriptorName/String")
            for term in el.iterfind("ConceptList/Concept/TermList/Term/String"):
                syn[term.text.lower()] = name
            syn[name.lower()] = name
            el.clear()
    return syn

class MeshTrie:
    def __init__(self, synonyms, min_single_len=4):
        self.root = {}
        for term, concept in synonyms.items():
            toks = tokenize(term)
            if not toks or (len(toks) == 1 and len(toks[0]) < min_single_len):
                continue  # skip very short single-word entries (abbreviation noise)
            node = self.root
            for t in toks: node = node.setdefault(t, {})
            node[END] = concept

    def extract(self, text):
        """Return Counter {concept: count}, scanning left to right, longest match first."""
        toks, i, out = tokenize(text), 0, Counter()
        while i < len(toks):
            node, j, best, best_j = self.root, i, None, i
            while j < len(toks) and toks[j] in node:
                node = node[toks[j]]; j += 1
                if END in node: best, best_j = node[END], j
            if best: out[best] += 1; i = best_j
            else: i += 1
        return out
