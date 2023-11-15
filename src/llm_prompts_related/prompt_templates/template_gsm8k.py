
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
# % Your task is to help me write {LANGUAGE} code. Below are the detailed instructions. Thanks for your help!

# % %%% Instructions Start

# % You will first be given one or more examples on writing {LANGUAGE} code about arithmetic problems.
# % You will then be given a new problem.
# % Please help me write {LANGUAGE} code about the new problem.

# % Here is a list of instructions that you should **strictly** follow when writing code:
# % (1) You should follow similar patterns from the given code examples and the following instructions.
# % (2) Your code should be accurate, self-consistent and complete.
# % (3) Use consistent variable names for coreferent entities or attributes across your code.
# % (4) Start by coding the given context after the "/* Context */" comment. Then code the query that represents the question after the "/* Query */" comment. 
# % (5) Please comment out the code about the question query with "%".

# % **Many thanks for your help! I am looking forward to your code!**

# % %%% Instructions End
# """

# INSTRUCTION_COT = \
# f"""
# Your task is to help me solve arithmetic reasoning problems. Below are the detailed instructions. Thanks for your help!

# **Instructions Start**

# You will first be given one or more examples about arithmetic problems.
# You will then be given a new problem.
# Please help me solve the new problem.

# Here is a list of instructions that you should **strictly** follow when writing code:
# (1) You should follow similar patterns from the given examples and the following instructions.
# (2) Your reasoning steps should be accurate, self-consistent and complete.
# (3) Your reasoning steps should follow the manner like "sent3 & sent4 -> int2: XXXX; sent-5 & int2 -> int3: XXXX; ...".
# (4) Please first write down the reasoning steps and then give me the answer. The answer should be in a new line, starting with "####".

# Many thanks for your help! I am looking forward to your solution!

# **Instructions End**
# """

################################################################################################################

# INSTRUCTION_PROLOG = \
# f"""

# Your task is to help me write **{LANGUAGE}** code. Below are the detailed instructions. Thanks for your help!

# **Instructions Start**

# (1) You should **strictly follow the following instructions**.
# (2) Your code should be accurate, self-consistent and complete. Use consistent variable names for coreferent entities or attributes throughout the code.
# (3) Start by coding the given context after the "/* Context */" comment. Then code the query that represents the question after the "/* Query */" comment. 
# (4) Please **only give me the Prolog code** and DO NOT show your reasoning steps in natural language.
# (5) DO NOT include any special symbols like "$" when representing numbers in Prolog, because such symbols cannot be processed by Prolog.

# Many thanks for your help! I am looking forward to your code!

# **Instructions End**

# """

INSTRUCTION_PROLOG = \
f"""

Could you please help me write {LANGUAGE} code to solve the following arithmetic reasoning problem? You should use consistent variable names for coreferent entities or attributes throughout the code. Start by coding the given context after the "/* Context */" comment. Then code the query that represents the question after the "/* Query */" comment. 

"""

INSTRUCTION_COT = \
f"""

Your task is to help me solve arithmetic reasoning problems. Below are the detailed instructions. Thanks for your help!

**Instructions Start**

(1) You should follow the following instructions.
(2) Your reasoning steps should be accurate, self-consistent and complete.
(3) Your reasoning steps should follow the manner like "sent3 & sent4 -> int2: XXXX; sent-5 & int2 -> int3: XXXX; ...".
(4) Please first write down the reasoning steps and then give me the answer. The answer should be in a new line, starting with "####".

Many thanks for your help! I am looking forward to your response!

**Instructions End**

"""

##########################################################################

DEMO_TEMPLATE_PROLOG = \
"""

% %%% Example Problem No.{} Start

{}


% %%% Reasoning Steps:

{}


% %%%% Code:

```prolog
{}
```

"""

DEMO_TEMPLATE_COT = \
"""

**Example Problem No.{} Start**

{}


**Reasoning Steps & Answer**

{}

"""

##########################################################################

# NEW_PROBLEM_TEMPLATE_PROLOG = \
# """

# % %%% New Problem Start

# {}


# """

# NEW_PROBLEM_TEMPLATE_COT = \
# """

# **New Problem Start**

# {}

# **Reasoning Steps & Answer**

# """