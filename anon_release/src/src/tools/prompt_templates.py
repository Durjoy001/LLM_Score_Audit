# src/tools/prompt_templates.py
# -*- coding: utf-8 -*-
"""
Reusable prompt template blocks for Stage 4 LLM answering.

Keep large prompt techniques here so llm_answering.py stays focused on:
- loading data
- calling providers
- parsing responses
- normalizing outputs
"""

def cot_reasoning_section() -> str:
    """
    Chain-of-Thought style prompt for proposal evaluation.
    Focus: Team, Objectives, Strategy, Advantages, Feasibility.
    """
    return """
CHAIN-OF-THOUGHT PROPOSAL EVALUATION PROMPT

You are evaluating a business, project, startup, research, or program proposal.

Think through the evaluation step by step before answering.
Do not reveal your full chain-of-thought.
Use your reasoning only to produce the final structured JSON output.

Step 1: Understand the proposal
Identify the type of proposal:
- business/startup
- SaaS/software
- hardware/product
- service business
- public program
- research project
- infrastructure project
- regulated product
- other

Identify the main problem, target users, and expected outcome.

Step 2: Evaluate Team
Reason about whether the team can execute the proposal.
Consider:
- relevant experience
- technical capability
- business or operational capability
- domain expertise
- missing roles
- advisors, partners, or governance support
- execution track record

Step 3: Evaluate Objectives
Reason about whether the objectives are clear and realistic.
Consider:
- problem-solution fit
- target users or beneficiaries
- measurable goals
- success indicators
- milestones
- scope realism

Step 4: Evaluate Strategy
Reason about whether the plan is practical and coherent.
Consider:
- implementation approach
- roadmap or timeline
- resource allocation
- go-to-market or adoption plan
- completed work versus future intention
- assumptions that still need validation

Step 5: Evaluate Advantages
Reason about whether the proposal has meaningful strengths.
Consider:
- unique value proposition
- innovation
- competitive edge
- intellectual property
- technology, data, partnerships, cost, timing, or location advantages
- whether each advantage is proven or only claimed

Step 6: Evaluate Feasibility
Reason about whether the proposal can realistically be executed.
Consider:
- technical practicality
- budget and funding needs
- available resources
- operational capacity
- legal, regulatory, or compliance issues
- key risks and dependencies

Step 7: Synthesize the evaluation
Compare the proposal facts against what would normally be expected for this type of proposal.
Separate:
- confirmed facts
- partial evidence
- plans or intentions
- missing evidence
- critical risks
- practical recommendations

Final output rules:
- Output only valid JSON.
- Do not reveal the chain-of-thought.
- Do not include these steps in the answer.
- Do not include examples.
- Do not add extra keys.
- Be precise and evidence-based.
- Prefer wording like: "The proposal states X, but does not provide evidence for Y."
- Do not say "no information exists" unless the proposal clearly does not include it.

Return this JSON structure only:

{
  "answer": [],
  "claims": [],
  "evidence_hints": [],
  "general_insights": [],
  "topic_tags": [],
  "confidence": 0.0,
  "caveats": []
}

Field requirements:
- "answer": Exactly 3–4 numbered points. STRICT FORMAT: each point must be a single line of 50–80 characters — do NOT wrap a point across multiple lines. Total answer: 160–300 characters. Example: "1. Team holds 3 propulsion patents.\n2. Revenue hit ¥100M last year.\n3. No cross-disciplinary PM detail shown."
- "claims": verifiable conclusions based only on proposal evidence.
- "evidence_hints": 2–5 short phrases or keywords copied VERBATIM from the proposal facts that directly support this answer. Must be exact text from the proposal — not section names, not paraphrases. Example: "年营收突破1亿元" not "财务信息部分提到的收入数据".
- "general_insights": general best-practice context only, not project-specific claims.
- "topic_tags": short lowercase tags.
- "confidence": number from 0.40 to 0.92. Use 0.40–0.54 when proposal facts are sparse or vague; 0.55–0.70 for moderate evidence; 0.71–0.92 when evidence is strong and specific.
- "caveats": missing information, assumptions, contradictions, or evaluation limits.
""".strip()


def react_reasoning_section() -> str:
    """
    ReAct-inspired proposal evaluation framework.
    Focus attributes:
    Team, Objective, Strategy, Advantages, Feasibility.

    This is not true external-tool ReAct.
    It guides the model to internally reason, check evidence, identify gaps,
    and produce only structured JSON.
    """
    return """
═══════════════════════════════════════════════════════════════════════════════
REACT-STYLE PROPOSAL EVALUATION FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════

You are evaluating a business, startup, project, research, program, product,
service, SaaS, infrastructure, or regulated proposal.

Use a ReAct-style internal reasoning process:
- Thought: understand what must be evaluated
- Action: inspect the provided proposal facts
- Observation: classify the evidence and gaps
- Final: produce the structured JSON only

IMPORTANT:
- Do NOT output Thought / Action / Observation.
- Do NOT reveal chain-of-thought.
- Do NOT claim external search, browsing, or tool use.
- Use only the provided proposal facts and allowed general industry knowledge.
- General industry knowledge must be clearly treated as general baseline, not as proof about the proposal.
- Be precise, evidence-based, and cautious.
- Prefer wording like: "The proposal states X, but does not provide evidence for Y."
- Do not say "no information exists" unless the provided proposal clearly lacks that information.

───────────────────────────────────────────────────────────────────────────────
INTERNAL REACT EVALUATION PROCESS
───────────────────────────────────────────────────────────────────────────────

STEP 1: Proposal Archetype Identification

Thought:
Determine what type of proposal is being evaluated.

Action:
Inspect the proposal facts for signals of:
- business/startup
- SaaS/software
- hardware/product
- service business
- marketplace
- public program
- research project
- infrastructure project
- regulated product
- other

Observation:
Identify the most likely archetype.
If unclear, state that the proposal does not fully reveal the archetype.

───────────────────────────────────────────────────────────────────────────────
STEP 2: Attribute-by-Attribute Evidence Check
───────────────────────────────────────────────────────────────────────────────

Evaluate the proposal using these five attributes only:

1. Team
2. Objective
3. Strategy
4. Advantages
5. Feasibility

For EACH attribute, internally perform the following ReAct-style checks:

Thought:
What should be evaluated for this attribute?

Action:
Inspect the provided proposal facts and separate evidence into:

- CONFIRMED:
  Explicitly stated facts with visible support, such as names, roles,
  experience, numbers, dates, milestones, results, documents, or clear claims.

- PARTIAL:
  Mentioned but lacking enough detail, proof, measurement, or explanation.

- PLANS:
  Future intentions, proposed actions, or goals without execution evidence.

- NOT VISIBLE:
  Information that is normally important but not shown in the provided facts.

Observation:
Compare the visible proposal evidence against what is normally expected
for this proposal archetype.

───────────────────────────────────────────────────────────────────────────────
ATTRIBUTE-SPECIFIC EVALUATION CRITERIA
───────────────────────────────────────────────────────────────────────────────

A. Team

Evaluate whether the team can realistically execute the proposal.

Consider:
- relevant founder/team experience
- technical capability
- business or operational capability
- domain expertise
- execution track record
- missing roles
- advisors, partners, or governance support
- hiring or outsourcing needs
- credibility of leadership

Evidence to look for:
- team bios
- CVs or LinkedIn profiles
- previous project history
- technical portfolio
- operational experience
- advisors or partners
- organizational chart
- role allocation

B. Objective

Evaluate whether the proposal objectives are clear, realistic, and measurable.

Consider:
- problem-solution fit
- target users, customers, or beneficiaries
- clear expected outcome
- measurable goals
- success indicators or KPIs
- milestones
- scope realism
- alignment between problem, solution, and impact

Evidence to look for:
- problem statement
- objective list
- target user definition
- measurable KPIs
- baseline data
- expected outputs and outcomes
- milestone timeline

C. Strategy

Evaluate whether the implementation plan is practical and coherent.

Consider:
- implementation approach
- roadmap or timeline
- resource allocation
- go-to-market or adoption plan
- customer acquisition or beneficiary outreach
- operational workflow
- completed work versus future intention
- assumptions that need validation
- dependency on partners, funding, regulation, or technology

Evidence to look for:
- roadmap
- launch plan
- marketing or adoption plan
- operational plan
- budget allocation
- project timeline
- pilot results
- user testing
- partnership plan
- risk mitigation plan

D. Advantages

Evaluate whether the proposal has meaningful strengths or defensible value.

Consider:
- unique value proposition
- innovation
- competitive edge
- intellectual property
- technology advantage
- data advantage
- partnerships
- cost advantage
- timing or location advantage
- market access
- whether each advantage is proven or only claimed

Evidence to look for:
- competitor comparison
- market research
- IP/patent/trademark proof
- exclusive partnerships
- proprietary technology
- user traction
- cost analysis
- testimonials
- pilot outcomes
- performance benchmarks

E. Feasibility

Evaluate whether the proposal can realistically be executed.

Consider:
- technical practicality
- financial viability
- budget and funding needs
- available resources
- operational capacity
- legal, regulatory, or compliance issues
- implementation risks
- market or adoption risks
- scalability
- sustainability
- key dependencies

Evidence to look for:
- financial projections
- cost breakdown
- funding plan
- technical architecture
- prototype or MVP
- compliance documents
- licenses or approvals
- operational capacity
- risk register
- mitigation plan
- scalability assumptions

───────────────────────────────────────────────────────────────────────────────
STEP 3: Gap and Risk Classification
───────────────────────────────────────────────────────────────────────────────

For each attribute, classify gaps as:

- CRITICAL:
  Could stop execution, funding, approval, launch, adoption, or sustainability.

- IMPORTANT:
  Significantly weakens credibility, execution quality, or stakeholder confidence.

- MINOR:
  Manageable weakness that can be fixed with additional detail or documentation.

Avoid broad absolute negatives.
Use careful wording:
- "The proposal states A, but does not show detailed evidence on B."
- "The proposal identifies X as a plan, but does not provide execution proof."
- "The proposal suggests Y, but the supporting data is only partial."
- "The proposal would be stronger with evidence such as Z."

Flag visible contradictions, unclear assumptions, or unsupported claims.

───────────────────────────────────────────────────────────────────────────────
STEP 4: Recommendation Synthesis
───────────────────────────────────────────────────────────────────────────────

For the strongest recommendations, identify:

- WHO should act
- WHAT should be done
- WHAT evidence should be produced
- WHY it matters for Team, Objective, Strategy, Advantages, or Feasibility

Prioritize recommendations by:
1. critical execution risk
2. stakeholder/investor confidence
3. measurable proof
4. practical next steps

───────────────────────────────────────────────────────────────────────────────
STEP 5: Final Output Mapping
───────────────────────────────────────────────────────────────────────────────

Map the internal evaluation into the final JSON as follows:

- "answer":
  Exactly 3–4 numbered points. STRICT FORMAT: each point must be a single
  line of 50–80 characters — do NOT wrap a point across multiple lines.
  Total answer: 160–300 characters.
  Example: "1. Team holds 3 propulsion patents.\n2. Revenue hit ¥100M.\n3. No PM detail shown."

- "claims":
  Verifiable conclusions based only on the provided proposal evidence.
  Include confirmed facts and carefully worded evidence gaps.

- "evidence_hints":
  2–5 short phrases or keywords copied VERBATIM from the proposal facts
  that directly support this answer. Must be exact text from the proposal
  — not section names, not paraphrases.
  Example: "年营收突破1亿元" not "财务信息部分提到的收入数据".

- "general_insights":
  General best-practice context only.
  Do not present these as project-specific facts.

- "topic_tags":
  3 to 6 short lowercase tags related to the proposal archetype,
  five attributes, risks, and evidence needs.

- "confidence":
  A number from 0.40 to 0.92. Use 0.40–0.54 when proposal facts are
  sparse or vague; 0.55–0.70 for moderate evidence; 0.71–0.92 when
  evidence is strong and specific.

- "caveats":
  Missing information, assumptions, contradictions, incomplete evidence,
  or evaluation limits.

───────────────────────────────────────────────────────────────────────────────
FINAL OUTPUT RULES
───────────────────────────────────────────────────────────────────────────────

Output ONLY valid JSON.

Do NOT include:
- Thought
- Action
- Observation
- chain-of-thought
- internal reasoning steps
- explanations outside JSON
- markdown
- examples
- extra keys

Return this JSON structure only:

{
  "answer": [],
  "claims": [],
  "evidence_hints": [],
  "general_insights": [],
  "topic_tags": [],
  "confidence": 0.0,
  "caveats": []
}

Field requirements:
- "answer": Exactly 3–4 numbered points, each on a SINGLE LINE of at most 60 characters. Total: 180–270 characters.
- "claims": verifiable conclusions based only on proposal evidence.
- "evidence_hints": 2–5 short phrases copied VERBATIM from the proposal facts (not paraphrases or section names).
- "general_insights": general best-practice context only.
- "topic_tags": short lowercase tags.
- "confidence": number from 0.40 to 0.92.
- "caveats": missing information, assumptions, contradictions, or limits.
""".strip()


def self_consistency_section() -> str:
    """
    Returns a Self-Consistency-inspired multi-perspective proposal evaluation framework.

    Focus attributes:
    Team, Objective, Strategy, Advantages, Feasibility.

    This is not full sampling-based self-consistency unless the application code
    makes multiple LLM calls and aggregates the outputs. This prompt version asks
    the model to internally compare several evaluation perspectives before
    producing the final JSON answer.
    """
    return """
═══════════════════════════════════════════════════════════════════════════════
SELF-CONSISTENCY-INSPIRED PROPOSAL EVALUATION FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════

You are evaluating a business, startup, project, research, program, product,
service, SaaS, infrastructure, or regulated proposal.

Use this as an internal multi-perspective checking framework only.

IMPORTANT:
- Do NOT output separate reasoning paths.
- Do NOT output voting notes.
- Do NOT output hidden reasoning.
- Do NOT mention that self-consistency was used.
- Do NOT claim external search, browsing, or tool use.
- Use only the provided proposal facts and allowed general industry knowledge.
- General industry knowledge must be treated only as general baseline context,
  not as proof about the proposal.
- Be precise, evidence-based, and cautious.
- Prefer wording like: "The proposal states X, but does not provide evidence for Y."
- Do not say "no information exists" unless the provided proposal clearly lacks it.

───────────────────────────────────────────────────────────────────────────────
CORE EVALUATION ATTRIBUTES
───────────────────────────────────────────────────────────────────────────────

Evaluate the proposal using these five attributes only:

1. Team
2. Objective
3. Strategy
4. Advantages
5. Feasibility

For each attribute, internally compare the proposal from multiple perspectives.
Each perspective should check the same five attributes from a different angle.
The final answer should synthesize the most consistent conclusions.

───────────────────────────────────────────────────────────────────────────────
PERSPECTIVE 1: FACTUAL AND EVIDENCE-BASED CHECK
───────────────────────────────────────────────────────────────────────────────

For Team:
- What names, roles, experience, advisors, partners, or execution history are visible?
- Is the team's capability confirmed, partially supported, only planned, or not visible?

For Objective:
- What problem, target users, expected outcomes, KPIs, milestones, or success indicators are visible?
- Are the objectives specific, measurable, and realistic?

For Strategy:
- What implementation plan, roadmap, timeline, resource plan, adoption plan, or go-to-market plan is visible?
- Does the proposal show completed work, or mostly future intention?

For Advantages:
- What unique value proposition, innovation, market edge, IP, partnership, cost, timing, location, or technology advantage is visible?
- Are advantages proven with evidence or only claimed?

For Feasibility:
- What budget, resources, technical proof, operational capacity, compliance support, licenses, approvals, or risk mitigation are visible?
- Can the proposal realistically be executed based on the provided facts?

Classify evidence as:
- CONFIRMED: explicit facts with visible support
- PARTIAL: mentioned but under-explained
- PLANS: future intention without execution proof
- NOT VISIBLE: normally important but not shown in the provided facts

───────────────────────────────────────────────────────────────────────────────
PERSPECTIVE 2: ATTRIBUTE QUALITY CHECK
───────────────────────────────────────────────────────────────────────────────

Evaluate the quality of each attribute:

Team:
- Is the team suitable for the proposal type?
- Are key technical, operational, financial, domain, and leadership roles covered?
- Are any important roles missing?

Objective:
- Is the objective clear, focused, and aligned with the problem?
- Is the target user/customer/beneficiary clearly defined?
- Are the expected outputs and outcomes measurable?

Strategy:
- Is the strategy practical and coherent?
- Does the plan connect resources, timeline, execution steps, and adoption?
- Are assumptions clearly stated and testable?

Advantages:
- Are the claimed strengths meaningful and defensible?
- Is there a clear difference from competitors or alternatives?
- Are the advantages backed by traction, data, IP, partnerships, or performance proof?

Feasibility:
- Is the proposal technically, financially, operationally, legally, and practically feasible?
- Are budget, implementation capacity, risks, and dependencies realistic?
- Is there enough evidence to believe the proposal can be delivered?

───────────────────────────────────────────────────────────────────────────────
PERSPECTIVE 3: RISK AND GAP CHECK
───────────────────────────────────────────────────────────────────────────────

For each of the five attributes, identify gaps and classify them as:

- CRITICAL:
  Could stop execution, funding, approval, launch, adoption, or sustainability.

- IMPORTANT:
  Significantly weakens credibility, execution quality, or stakeholder confidence.

- MINOR:
  Manageable weakness that can be fixed with additional detail or documentation.

Check specifically:

Team risks:
- missing execution skills
- unclear leadership capacity
- no technical or domain proof
- weak governance or advisor support

Objective risks:
- vague problem definition
- unclear target users
- non-measurable goals
- unrealistic scope

Strategy risks:
- unclear implementation roadmap
- weak adoption or go-to-market plan
- resource gaps
- untested assumptions
- dependency on uncertain partners or funding

Advantages risks:
- generic value proposition
- unsupported innovation claims
- weak competitor comparison
- advantages that are claimed but not proven

Feasibility risks:
- unclear budget
- lack of prototype, pilot, or technical proof
- operational capacity gaps
- regulatory or compliance uncertainty
- scalability or sustainability concerns

Use careful wording:
- "The proposal states A, but does not show detailed evidence on B."
- "The proposal identifies X as a plan, but does not provide execution proof."
- "The proposal suggests Y, but the supporting data is partial."
- "The proposal would be stronger with evidence such as Z."

Do not use broad absolute negatives unless clearly supported by the provided facts.

───────────────────────────────────────────────────────────────────────────────
PERSPECTIVE 4: STAKEHOLDER-READINESS CHECK
───────────────────────────────────────────────────────────────────────────────

Evaluate how ready the proposal is for investors, funders, procurement teams,
partners, operators, academic reviewers, or decision-makers.

For Team:
- Would stakeholders trust this team to execute?
- What proof of capability would they expect?

For Objective:
- Would stakeholders understand exactly what the proposal aims to achieve?
- Are the success indicators strong enough?

For Strategy:
- Would stakeholders see a realistic execution path?
- Are timeline, budget, adoption, and operational steps convincing?

For Advantages:
- Would stakeholders see a clear reason this proposal is better than alternatives?
- Are the claimed strengths defensible?

For Feasibility:
- Would stakeholders believe the proposal can be implemented with available
  resources, budget, technology, approvals, and risk controls?

Keep general stakeholder expectations separate from project-specific facts.

───────────────────────────────────────────────────────────────────────────────
PERSPECTIVE 5: ACTION AND RECOMMENDATION CHECK
───────────────────────────────────────────────────────────────────────────────

Identify the top actions that would most improve proposal readiness.

For each recommendation, internally identify:
- WHO should act
- WHAT should be done
- WHAT evidence/output should be produced
- WHICH attribute it strengthens:
  Team, Objective, Strategy, Advantages, or Feasibility
- WHY it matters

Prioritize recommendations by:
1. critical execution risk
2. stakeholder confidence
3. measurable proof
4. practical next steps

Examples of evidence/output types:
- team bios or CVs
- role allocation chart
- technical architecture
- prototype or MVP
- pilot results
- user/customer validation
- market research
- competitor comparison
- budget breakdown
- financial projections
- implementation roadmap
- risk register
- compliance documents
- licenses or approvals
- partnership letters
- KPI dashboard
- monitoring and evaluation framework

───────────────────────────────────────────────────────────────────────────────
INTERNAL CONSISTENCY CHECK
───────────────────────────────────────────────────────────────────────────────

Compare conclusions across all perspectives.

- If multiple perspectives support the same conclusion, treat it as stronger.
- If perspectives disagree, reflect uncertainty in the answer and caveats.
- Do not force a simple yes/no conclusion when the evidence supports a mixed result.
- Prefer balanced conclusions such as:
  "partially ready",
  "credible but under-evidenced",
  "promising but dependent on X/Y proof",
  "strategically clear but operationally incomplete",
  or "feasible only if key assumptions are validated."

The final evaluation must remain focused on:
- Team
- Objective
- Strategy
- Advantages
- Feasibility

───────────────────────────────────────────────────────────────────────────────
OUTPUT MAPPING
───────────────────────────────────────────────────────────────────────────────

Map the internal multi-perspective evaluation into the final JSON as follows:

- "answer":
  Exactly 3–4 numbered points. STRICT FORMAT: each point must be a single
  line of 50–80 characters — do NOT wrap a point across multiple lines.
  Total answer: 160–300 characters.
  Example: "1. Team holds 3 propulsion patents.\n2. Revenue hit ¥100M.\n3. No PM detail shown."

- "claims":
  Verifiable proposal-based conclusions and carefully worded evidence gaps.
  Use only the provided proposal facts.

- "evidence_hints":
  2–5 short phrases or keywords copied VERBATIM from the proposal facts
  that directly support this answer. Must be exact text from the proposal
  — not section names, not paraphrases.
  Example: "年营收突破1亿元" not "财务信息部分提到的收入数据".

- "general_insights":
  General best-practice context only.
  Do not present these as project-specific facts.

- "topic_tags":
  3 to 6 short lowercase tags based on the proposal archetype,
  five attributes, risks, and evidence needs.

- "confidence":
  A number from 0.40 to 0.92 based on:
  - completeness of proposal facts (sparse → 0.40–0.54, moderate → 0.55–0.70, strong → 0.71–0.92)
  - consistency across perspectives
  - strength of visible evidence
  - number of unresolved gaps

- "caveats":
  Missing information, assumptions, contradictions, evaluation limits,
  or areas where the evidence supports only a partial conclusion.

───────────────────────────────────────────────────────────────────────────────
FINAL OUTPUT CONTROL
───────────────────────────────────────────────────────────────────────────────

Output ONLY valid JSON.

Do NOT include:
- internal perspectives
- reasoning paths
- voting tables
- self-consistency notes
- hidden reasoning
- examples
- markdown
- explanations outside JSON
- extra keys

Return this JSON structure only:

{
  "answer": [],
  "claims": [],
  "evidence_hints": [],
  "general_insights": [],
  "topic_tags": [],
  "confidence": 0.0,
  "caveats": []
}

Field requirements:
- "answer": Exactly 3–4 numbered points, each on a SINGLE LINE of at most 60 characters. Total: 180–270 characters.
- "claims": verifiable conclusions based only on proposal evidence.
- "evidence_hints": 2–5 short phrases copied VERBATIM from the proposal facts (not paraphrases or section names).
- "general_insights": general best-practice context only.
- "topic_tags": short lowercase tags.
- "confidence": number from 0.40 to 0.92.
- "caveats": missing information, assumptions, contradictions, uncertainty, or limits.
""".strip()