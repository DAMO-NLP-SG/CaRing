
LANGUAGE = "Prolog"
# COMMENT_SYMBOL = "%"

##########################################################################

PROMPT_TEMPLATE = \
"""
{}

{}

{}


"""

##########################################################################

# INSTRUCTION_PROLOG = \
# f"""
# Your task is to help me write **{LANGUAGE}** code. Below are the detailed instructions. Thanks for your help!

# **Instructions Start**

# You will first be given one or more examples on writing {LANGUAGE} code about logical reasoning problems. 
# You will then be given a new problem.
# Please help me write {LANGUAGE} code about the new problem.

# Here is a list of instructions that you should **strictly** follow when writing code:
# (1) You should **strictly follow similar patterns from the given code examples and the following instructions**.
# (2) Your code should be accurate, self-consistent and complete. Use consistent variable names for coreferent entities or attributes across all triples and rules.
# (3) Start by coding the triples after the "/* Triples */" comment. Then code the rules after the "/* Rules */" comment. Finally, code the question statements after the "/* Questions */" comment. Please comment out the code about the question statements with "%".
# (4) Please only give me the Prolog code. Please ignore any logic conflicts or errors in the problem.

# Many thanks for your help! I am looking forward to your code!

# **Instructions End**

# """

# INSTRUCTION_COT = \
# f"""
# Your task is to help me solve logic reasoning problems. Below are the detailed instructions. Thanks for your help!

# **Instructions Start**

# You will first be given one or more examples about logic reasoning problems. You will then be given a new problem. Please help me solve the new problem.

# Here is a list of instructions that you should **strictly** follow when writing code:
# (1) You should follow similar patterns from the given examples and the following instructions.
# (2) Your reasoning steps should be accurate, self-consistent and complete.
# (3) Your reasoning steps should follow the manner like "triple-1 & rule-4 -> int-1: XXXX; triple-5 & int-1 -> int-2: XXXX; ...".
# (4) Please first write down the reasoning steps and then give me the answer. The answer should be in a new line, starting with "####". The answer should be among "True", "False" and "Unknown".

# Many thanks for your help! I am looking forward to your response!

# **Instructions End**

# """

INSTRUCTION_PROLOG = \
f"""
Your task is to help me write **{LANGUAGE}** code. Below are the detailed instructions. Thanks for your help!

**Instructions Start**

(1) You should **strictly follow the following instructions**.
(2) Your code should be accurate, self-consistent and complete. Use consistent variable names for coreferent entities or attributes across all triples and rules.
(3) Start by coding the triples after the "/* Triples */" comment. Then code the rules after the "/* Rules */" comment. Finally, code the question statements after the "/* Questions */" comment.
(4) Please **only give me the Prolog code** and DO NOT show your reasoning steps in natural language.

Many thanks for your help! I am looking forward to your code!

**Instructions End**

"""

INSTRUCTION_COT = \
f"""
Your task is to help me solve logic reasoning problems. Below are the detailed instructions. Thanks for your help!

**Instructions Start**

(1) You should follow the following instructions.
(2) Your reasoning steps should be accurate, self-consistent and complete.
(3) Your reasoning steps should follow the manner like "triple-1 & rule-4 -> int-1: XXXX; triple-5 & int-1 -> int-2: XXXX; ...".
(4) Please first write down the reasoning steps and then give me the answer. The answer should be in a new line, starting with "####". The answer should be among "True", "False" and "Unknown".
(5) If you believe the answer is "Unknown", then you do not need to give reasoning steps.

Many thanks for your help! I am looking forward to your response!

**Instructions End**

"""

INSTRUCTION_DIRECT = \
f"""
Your task is to help me solve logic reasoning problems. Please write down your answer after "####". You answer should be among ("True", "False", "Unknown"). 

Many thanks for your help! I am looking forward to your response!

"""

##########################################################################

DEMO_TEMPLATE_PROLOG = \
"""

### Example Problem No.{} Start

{}

**Code**:

```prolog
{}
```

### Example Problem No.{} End

"""

DEMO_TEMPLATE_COT = \
"""

### Example Problem No.{} Start

{}

**Reasoning Steps & Answer**:

{}

### Example Problem No.{} End

"""

DEMO_TEMPLATE_DIRECT = \
"""

### Example Problem No.{} Start

{}

**Answer**:

{}

### Example Problem No.{} End

"""

##########################################################################

# NEW_PROBLEM_TEMPLATE_PROLOG = \
# """

# ### New Problem Start

# Triples:
# {}

# Rules:
# {}

# Question:
# Whether the following statement is True or False or Uncertain?
# {}

# **Code**:

# """

# NEW_PROBLEM_TEMPLATE_COT = \
# """

# ### New Problem Start

# Triples:
# {}

# Rules:
# {}

# Question:
# Whether the following statement is True or False or Uncertain?
# {}

# **Reasoning Steps & Answer**:

# """

NEW_PROBLEM_TEMPLATE_PROLOG = \
"""
Triples:
{}

Rules:
{}

Question:
Whether the following statement is True or False or Uncertain?
{}

"""

NEW_PROBLEM_TEMPLATE_COT = \
"""
Triples:
{}

Rules:
{}

Question:
Whether the following statement is True or False or Uncertain?
{}

"""