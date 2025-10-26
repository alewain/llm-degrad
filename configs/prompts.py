"""
Prompt lists for the four main experiments (all instruction-tuned).

This module contains the prompts used in the thesis experiments:
- dream_prompts_it: Dream narration task (~38 prompts)
- math_prompts_it: Math tasks (~24 prompts)
- language_prompts_it: Language tasks (~16 prompts)
- cookie_theft_prompts_it: Cookie Theft image description (~20 prompts)

Note: Only instruction-tuned (IT) variants are included (hence the `_it` suffix in names).
      Pretrained (PT) variants are not part of this version.
"""

# ============================================================================
# Dream narration prompts (dreams_it experiment)
# ============================================================================

dream_prompts_it = [
    "Just narrate a dream with as much detail as you can.",
    "Just describe a dream as vividly and narratively as possible.",
    "Just narrate a dream.",
    "Just describe a dream.",
    "Tell me about a dream.",
    "Describe a dream you had.",
    "Write down a dream.",
    "Share a dream.",
    "A dream — write it.",
    "Tell a dream in as much detail as you can.",
    "Describe a dream with as many details as possible.",
    "Narrate a dream clearly and fully.",
    "Write a dream and make it vivid.",
    "Describe a dream scene by scene.",
    "Write a dream with all the details you remember.",
    "Describe what happened in a dream.",
    "Tell a dream in full.",
    "Describe a dream from beginning to end.",
    "Write about what you saw in a dream.",
    "Tell what happened in a dream, with detail.",
    "Describe the scenes of a dream.",
    "Share a dream with as much clarity as possible.",
    "Tell a dream like you're telling a story.",
    "Write a dream and include what you felt.",
    "Tell what you experienced in a dream.",
    "Describe a dream like you lived it.",
    "Just write a text that begins with: 'I dreamed this recently.'",
    "Just write a text that begins with: 'I dreamed this recently. I'm going to recall the dream as clearly as I can.'",
    "Just write a text that begins with: 'Most people start the day by'",
    "Just write a text that begins with: 'Today I am feeling'",
    "Just write a text that begins with: 'The thing I like most in the world is'",
    "Just write a text that begins with: 'When I was a little kid'",
    "Just write a text that begins with: 'I had a terrifying dream last night in which'",
    "Just write a text that begins with: 'I worry a lot about'",
    "Just write a text that begins with: 'I will now name as many words related to the kitchen as I can:'",
    "Just write a text that begins with: 'I had a dream last night in which'",
    "Just write a text that begins with: 'I had a dream recently in which'",
    "Just write a text that begins with: 'I dreamed that'",
]

# ============================================================================
# IQ prompts (multi-task cognitive assessment)
# ============================================================================

# Math problems (instruction-tuned)
math_prompts_it = [
    "What number completes the sequence: 93, 94, 95, __, 97?",
    "What number continues the sequence: -5, -4, -3, -2, __?",
    "What is the result of 3 + 48 / 6 - 7?",
    "A train carriage seats 6 passengers. If three identical carriages are joined together, how many passengers can they seat in total?",
    "What number comes next in the sequence: 1.2, 4.2, 7.2, __?",
    "What number comes next in the sequence: 120, 113, 106, __?",
    "What number comes next in the sequence: 36, 49, 64, 81, __?",
    "What number comes next in the sequence: 4.5, 3.0, 1.5, __?",
    "What is the probability of randomly selecting a red item from a box containing 5 blue objects, 13 red ones, and 5 yellow ones?",
    "What number comes next in the sequence: 9.25, 7.5, 5.75, __?",
    "What number is missing in the arithmetic sequence: 99, __, 71, 57?",
    "What number makes the equation true: 12 × 5 = 15 × __?",
    "What number comes next in the sequence: 4, 5, 7, 10, 14, __?",
    "A charity raised £84 in two days. On Monday it raised £12 more than on Tuesday. How much was raised on Tuesday?",
    "A rectangle has a length that is 5 times its width. If its perimeter is 72 cm, what is its area?",
    "What number makes the equation true: 3(x - 5) = 4x + 7?",
    "A car travels 150 km in 2 hours and 30 minutes. What is its average speed in km/h?",
    "The sum of three consecutive odd numbers is 111. What is the smallest of these numbers?",
    "A water tank is 4/7 full. After removing 36 litres, it is 2/7 full. What is the full capacity of the tank in litres?",
    "In a class of 40 students, 60% are girls. If 50% of the boys and 25% of the girls wear glasses, how many students wear glasses?",
    "What is the smallest positive integer greater than 100 that is divisible by both 6 and 15?",
    "A jacket originally priced at £80 is sold with a 25% discount. What is the sale price in pounds?",
    "If 3/4 of a number is 27, what is the number?",
    "Tom has 3 red balls, 2 blue balls, and 5 green balls. He also has a dog named Max. How many balls does Tom have in total?",
    "How many hours are there in four and a half days?",
]

# Language tasks (instruction-tuned)
language_prompts_it = [
    "Which word comes next: January, March, May, ___?",
    "Which word does not belong: carrot, apple, banana, grape",
    "Which word does not belong: spoon, tea, coffee, juice",
    "Find the odd one out: walk, blue, run, jump",
    "Find the odd one out: red, green, square, blue",
    "Select the word that is different: star, moon, chair, sun",
    "Which word does not belong: prism, sphere, pyramid, circle",
    "Rearrange the following to form a correct sentence: 'window / the / through / breeze / gentle / a / came'",
    "Rearrange the following to form a correct sentence: 'books / of / pile / a / table / the / on / rested'",
    "Find the antonym of 'scarce' among these: abundant, limited, rare, minimal",
    "Rearrange the following to form a correct sentence: 'abandoned / was / lighthouse / fog / thick / the / by / the / hidden'",
    "Which word is closest in meaning to 'enormous'? A) huge B) small C) narrow D) little",
    "Choose the word that does not fit: cello, violin, hammer, trumpet",
    "Which of these is different: shout, orbit, spin, rotate",
    "Rearrange the following to form a correct sentence: 'beneath / quietly / bridge / the / river / the / flowed'",
]

# Logic puzzles (instruction-tuned) - unused by default tasks
logic_prompts_it = [
    "All cats are animals. All animals are floms. Can we conclude that all cats are floms?",
    "All cats are animals. All animals are floms. Can we conclude that all floms are cats?",
    "If the statement 'If it rains, the grass is wet' is true, and the grass is not wet, what can we conclude about rain?",
]

# Factual knowledge (instruction-tuned) - unused by default tasks
factual_prompts_it = [
    "How many sides does a hexagon have?",
    "What is the capital of France?",
    "How many hours are there in two days?",
    "Which planet is known as the Red Planet?",
    "What gas do plants absorb during photosynthesis?",
]

# Creativity tasks (instruction-tuned) - unused by default tasks
creativity_prompts_it = [
    "List as many animals as you can.",
    "Name as many words related to the kitchen as you can.",
]

# Combined IQ prompts (all categories) - deprecated (kept for reference)
iq_prompts_it = (
    math_prompts_it + 
    language_prompts_it + 
    logic_prompts_it + 
    factual_prompts_it + 
    creativity_prompts_it
)

# ============================================================================
# Cookie Theft image description prompts
# ============================================================================

cookie_theft_prompts_it = [
    "Tell me everything you see going on in this picture.",
    "Tell me everything you see going on in this picture. Describe everything that is happening.",
    "Describe everything happening in this picture.",
    "Tell me what you see going on in this scene.",
    "List everything you see in the image.",
    "Describe in detail everything that is happening in the image.",
    "Describe this picture in as much detail as possible.",
    "Provide a complete description of the image.",
    "Tell me everything you see happening within this scene.",
    "Give a thorough account of the visual information presented in the drawing.",
    "Recount the entire contents of the image.",
    "Offer a comprehensive summary of what is depicted in this illustration.",
    "Present an exhaustive description of the picture's content.",
    "Please detail all the elements and activities shown in the picture.",
    "Articulate everything that is visibly occurring in this drawing.",
    "Describe this picture.",
    "Describe what you see in this image.",
    "Provide a detailed description of the scene shown.",
    "Give a comprehensive account of everything depicted in this picture.",
    "Describe the contents of this image in detail.",
]

