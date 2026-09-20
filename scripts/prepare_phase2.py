"""Freeze replication cohorts before any new model outputs are inspected."""
import hashlib
import json
import random
from pathlib import Path
from benchmark import read_rows
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'phase2/data'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'evaluation.jsonl').exists():
        raise SystemExit('Phase 2 cohorts already frozen')
    rust=read_rows(ROOT/'data/heldout.jsonl')
    random.Random(20260918).shuffle(rust)
    rows=[]
    for i,r in enumerate(rust[:64]):
        rows.append({**r,'task':'rust_seen' if i<32 else 'rust_fresh'})
    python=read_rows(ROOT/'data/humaneval-python.jsonl')
    random.Random(20260918).shuffle(python)
    rows.extend({**r,'task':'python'} for r in python[:32])
    controls={
      'prose':[
        'Explain how a bicycle gear ratio affects pedaling on a hill, in about 120 words.',
        'Write a 120-word story about a librarian finding a note in a returned book.',
        'Explain why urban trees can reduce summer heat, in about 120 words.',
        'Write a polite 100-word email asking a landlord to repair a leaking kitchen tap.',
        'Compare learning piano alone and with a teacher, in about 120 words.',
        'Describe a practical plan for organizing a small community book exchange, in 120 words.',
        'Explain the difference between weather and climate to a teenager, in 120 words.',
        'Write a 120-word museum label about the invention of the printing press.'
      ],
      'reasoning':[
        'A cyclist rides 18 km at 12 km/h and 18 km at 18 km/h. What is the average speed for the trip? Explain briefly.',
        'Three boxes are labeled apples, oranges, and mixed. Every label is wrong. You may draw one fruit from one box. Explain how to relabel all three.',
        'A tank fills in 6 hours with one pipe and empties in 9 hours with another. Starting empty with both open, how long until full? Show the calculation.',
        'If every wug is a dax and no dax is a zed, can a wug be a zed? Explain the logic.',
        'A fair coin is tossed four times. What is the probability of exactly two heads? Explain.',
        'A shop discounts a price by 20 percent and then increases the discounted price by 20 percent. Is it back at the original price? Use a numerical example.',
        'Five people each shake hands once with every other person. How many handshakes are there? Explain without double counting.',
        'A train leaves at 09:20 and travels 135 km at 90 km/h without stops. At what time does it arrive? Show the steps.'
      ],
      'tool_calls':[
        'Return only a JSON object with keys tool and arguments. Available tool: search_books(query: string, limit: integer). Find three books about ocean navigation.',
        'Return only a JSON object with keys tool and arguments. Available tool: convert_units(value: number, from: string, to: string). Convert 15 kilometers to miles.',
        'Return only a JSON object with keys tool and arguments. Available tool: weather(city: string, units: string). Get the weather in Kyoto in Celsius.',
        'Return only a JSON object with keys tool and arguments. Available tool: search_files(query: string, extension: string). Find PDF files about quarterly planning.',
        'Return only a JSON object with keys tool and arguments. Available tool: translate(text: string, target_language: string). Translate Good morning into Spanish.',
        'Return only a JSON object with keys tool and arguments. Available tool: calculate(expression: string). Calculate the square root of 144 plus 7.',
        'Return only a JSON object with keys tool and arguments. Available tool: lookup_word(word: string, language: string). Define serendipity in English.',
        'Return only a JSON object with keys tool and arguments. Available tool: search_recipes(ingredients: array of strings, vegetarian: boolean). Find vegetarian recipes with lentils and carrots.'
      ]}
    for domain,prompts in controls.items():
        rows.extend({'id':f'phase2-{domain}-{i}','task':domain,'prompt':p,'source':'locally authored diagnostic; not a standardized benchmark'} for i,p in enumerate(prompts))
    target=OUT/'evaluation.jsonl'
    target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (OUT/'manifest.json').write_text(json.dumps({'count':len(rows),'cohorts':{'rust_seen':32,'rust_fresh':32,'python':32,'prose':8,'reasoning':8,'tool_calls':8},'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'note':'Frozen before phase 2 outputs; diagnostic controls are small and not capability benchmarks.'},indent=2))
if __name__=='__main__':main()
