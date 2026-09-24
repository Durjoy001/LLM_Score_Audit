"""
Manual review of the MINUS edits (written by Claude, no LLM API calls).

REMOVED text
------------
For each proposal x sub-rubric, the passages removed are the sentences in the scorer's window (the first 12,000
characters of the generated report) that carry evidence FOR a higher score on that sub-rubric:
  doc-verifiable : the factual passage (names, figures, patents, budgets, milestones)
  judgment-dep   : the passage giving the signals or assessment an evaluator would weigh
Concerns, recommendations and boilerplate are never removed: they support a LOW score, so removing them cannot
test whether the score falls. A sentence is assigned to at most one sub-rubric within a dimension, because all
8 sub-rubrics of a dimension are removed together in one MINUS report.

A sub-rubric missing from MINUS_REMOVE[pid][dim] has NO supporting passage in the window: nothing to remove.
Those MINUS checks cannot test anything and should be reported separately from real checks.

Every occurrence of a passage in the report must be replaced (the executive summary repeats the commentary);
the original adaptive_strip_minus replaced only the first occurrence.

REPLACED WITH
-------------
One neutral sentence per sub-rubric (MINUS_REPLACEMENT). It states only that the proposal does not describe
something, never what that absence implies about the venture.
  Bad : "No competitive moat exists."
  Good: "The proposal does not describe competitive barriers or differentiation from competitors."
"""

MINUS_REPLACEMENT = {
    # strategy
    "strategy_R1_gtm": "The proposal does not describe a sales or distribution channel.",
    "strategy_R2_milestones": "The proposal does not describe a roadmap or dated milestones.",
    "strategy_R3_revenue": "The proposal does not describe pricing or a revenue model.",
    "strategy_R4_regulatory": "The proposal does not describe regulatory requirements or an approval pathway.",
    "strategy_R5_moat": "The proposal does not describe competitive barriers or differentiation from competitors.",
    "strategy_R6_partners": "The proposal does not name partners or describe any partner engagement.",
    "strategy_R7_team_fit": "The proposal does not describe the team's relevant background or prior experience.",
    "strategy_R8_timing": "The proposal does not discuss market timing or the entry window.",
    # objectives
    "objectives_R1_problem_clarity": "The proposal does not describe the problem or its context.",
    "objectives_R2_market_size": "The proposal does not give a market size.",
    "objectives_R3_buyer_pathway": "The proposal does not identify buyers or payers.",
    "objectives_R4_unmet_need_evidence": "The proposal does not cite evidence of an unmet need.",
    "objectives_R5_why_now": "The proposal does not discuss why the project should start now.",
    "objectives_R6_competitive_context": "The proposal does not mention existing solutions or competitors.",
    "objectives_R7_problem_solution_fit": "The proposal does not explain how the solution relates to the problem.",
    "objectives_R8_addressability": "The proposal does not define a target segment or how it will be reached.",
    # advantages
    "advantages_R1_mechanism_novelty": "The proposal does not describe the technical mechanism or how it differs from existing approaches.",
    "advantages_R2_ip_status": "The proposal does not mention patents or other intellectual property.",
    "advantages_R3_performance_proof": "The proposal does not report performance data.",
    "advantages_R4_competitor_benchmark": "The proposal does not compare the technology with competing products.",
    "advantages_R5_defensibility": "The proposal does not describe barriers that would prevent others from copying the technology.",
    "advantages_R6_platform_potential": "The proposal does not describe other applications of the technology.",
    "advantages_R7_validation_signals": "The proposal does not mention external validation such as publications, regulatory milestones or paying customers.",
    "advantages_R8_adoption_readiness": "The proposal does not mention pilots, letters of intent or customer contracts.",
    # team
    "team_R1_composition": "The proposal does not list team members or their roles.",
    "team_R2_credentials": "The proposal does not describe team members' qualifications.",
    "team_R3_track_record": "The proposal does not describe the team's prior projects or ventures.",
    "team_R4_governance": "The proposal does not describe a board, advisors or decision-making structure.",
    "team_R5_capacity": "The proposal does not state the team's size or members' time commitment.",
    "team_R6_complementarity": "The proposal does not describe the mix of skills across the team.",
    "team_R7_cohesion": "The proposal does not say whether team members have worked together before.",
    "team_R8_key_person_risk": "The proposal does not describe how responsibilities are shared or who could cover key roles.",
    # feasibility
    "feasibility_R1_budget_detail": "The proposal does not provide a budget or budget breakdown.",
    "feasibility_R2_funding_secured": "The proposal does not describe funding sources or committed funding.",
    "feasibility_R3_infrastructure": "The proposal does not describe facilities, equipment or data access.",
    "feasibility_R4_technical_readiness": "The proposal does not describe a prototype, pilot or test results.",
    "feasibility_R5_risk_mitigation": "The proposal does not discuss project risks or how they would be handled.",
    "feasibility_R6_resource_timeline_fit": "The proposal does not describe a timeline or the resources assigned to it.",
    "feasibility_R7_financial_realism": "The proposal does not compare the budget with similar projects or state how long the funds will last.",
    "feasibility_R8_sustainability": "The proposal does not describe how operations would be funded after the current funding.",
}

MINUS_REMOVE = {
    "p1": {
        "strategy": {
            "strategy_R5_moat": ["The patent application for the heart failure large model indicates a commitment to innovation and intellectual property protection."],
            "strategy_R6_partners": ["Collaboration with multiple hospitals provides access to clinical data and enhances the project's practical relevance."],
            "strategy_R7_team_fit": ["The team possesses strong expertise in cardiovascular medicine and data science, which is crucial for the project's success in heart failure management."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["The project addresses a critical need for standardized treatment for over 12 million heart failure patients in China."],
            "objectives_R7_problem_solution_fit": [
                "The project objectives well with the urgent need for standardized heart failure treatment in China, targeting significant patient outcomes like reducing readmission rates.",
                "Clear goals such as reducing regional disparities in diagnosis and treatment demonstrate a strong with patient needs.",
            ],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": [
                "The project introduces innovative technologies such as a dynamic replication mechanism and an ER compression algorithm, which could enhance data transmission efficiency.",
                "The use of a transformer-based multi-agent system indicates a sophisticated approach to heart failure management.",
            ],
            "advantages_R2_ip_status": ["The patent application for the heart failure large model indicates a commitment to innovation and intellectual property protection."],
            "advantages_R8_adoption_readiness": ["The strategy includes plans for training and technical support, which are critical for successful implementation."],
        },
        "team": {
            "team_R2_credentials": ["The team includes experts like Professor Zhang Liang, whose extensive surgical experience enhances credibility and clinical direction."],
            "team_R6_complementarity": ["The team possesses strong expertise in cardiovascular medicine and data science, which is crucial for the project's success in heart failure management."],
        },
        "feasibility": {
            "feasibility_R1_budget_detail": ["A comprehensive budget plan indicates a thorough approach to financial planning for construction, personnel, and operational expenses."],
            "feasibility_R3_infrastructure": [
                "The establishment of provincial and municipal data centers is a strategic move to ensure long-term resource stability.",
                "Collaboration with multiple hospitals provides access to clinical data and enhances the project's practical relevance.",
            ],
            "feasibility_R5_risk_mitigation": ["Existing discussions on risk assessment tools show an awareness of potential challenges."],
        },
    },
    "p2": {
        "strategy": {
            "strategy_R2_milestones": ["Key milestones are aligned with the project's primary objectives, indicating a structured approach."],
            "strategy_R5_moat": ["The focus on molecular modification design methods presents a novel approach to enhancing enzyme performance."],
            "strategy_R6_partners": ["Collaborations with established organizations may enhance innovation capacity and resource access."],
            "strategy_R7_team_fit": ["Leadership experience of Dr. Zhang Lu Jia and Dr. Feng Yinghui enhances credibility and aligns with project goals."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": [
                "The project's objectives are focused on developing high-performance enzyme products to address significant biological problems.",
                "The project has clearly defined goals centered on enzyme product development without unrelated sub-goals.",
            ],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": [
                "The focus on molecular modification design methods presents a novel approach to enhancing enzyme performance.",
                "The project claims significant advancements in enzyme technology, particularly in protease activity and enzyme design efficiency.",
            ],
            "advantages_R3_performance_proof": ["The project demonstrates a two-fold improvement in protease activity, indicating significant innovation."],
            "advantages_R6_platform_potential": ["The proposal outlines potential applications in various industries, suggesting a broad market impact."],
        },
        "team": {
            "team_R1_composition": ["Leadership experience of Dr. Zhang Lu Jia and Dr. Feng Yinghui enhances credibility and aligns with project goals."],
            "team_R2_credentials": ["The team's strong publication record supports their expertise and potential for attracting partnerships."],
            "team_R6_complementarity": ["The team demonstrates a strong foundation in academic and industry experience, particularly in enzyme technology and molecular therapy."],
        },
        "feasibility": {
            "feasibility_R1_budget_detail": [
                "The project outlines a funding allocation plan that prioritizes platform improvement and production optimization, which are critical for market development.",
                "The funding allocation plan supports both technology enhancement and operational efficiency.",
            ],
            "feasibility_R3_infrastructure": ["Collaborations with established organizations may enhance innovation capacity and resource access."],
            "feasibility_R8_sustainability": ["The structured investment approach suggests a long-term vision for addressing market needs."],
        },
    },
    "p3": {
        "strategy": {
            "strategy_R5_moat": [
                "Innovative approaches to antibody design may provide a competitive edge in the market.",
                "The integration of advanced protein design techniques could provide a unique market position.",
            ],
            "strategy_R6_partners": [
                "The project outlines a strategy that includes collaboration with the Institute of Biophysics to mitigate market risks and enhance technical capabilities.",
                "Collaboration with a reputable research institution may enhance technical support and market credibility.",
            ],
            "strategy_R7_team_fit": ["The leadership of Professor Wang Jiangyun adds academic credibility and enhances execution potential."],
            "strategy_R8_timing": ["The project's goals are aligned with urgent industry needs, indicating strong market potential."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["The project's objectives well with industry demands for functional antibody binding proteins, particularly in enhancing purification efficiency and reducing production costs."],
            "objectives_R5_why_now": ["The project's goals are aligned with urgent industry needs, indicating strong market potential."],
            "objectives_R6_competitive_context": ["Innovative approaches to antibody design may provide a competitive edge in the market."],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": [
                "The project proposes a novel protein design platform aimed at improving antibody binding proteins.",
                "The development of a proprietary AI model to accelerate R&D processes could significantly reduce development timelines.",
            ],
            "advantages_R2_ip_status": ["The team's extensive publication record and patent portfolio indicate a solid foundation in innovation and research capabilities."],
            "advantages_R5_defensibility": [
                "The integration of advanced protein design techniques could provide a unique market position.",
                "Innovative approaches to antibody design may provide a competitive edge in the market.",
            ],
        },
        "team": {
            "team_R1_composition": ["The leadership of Professor Wang Jiangyun adds academic credibility and enhances execution potential."],
            "team_R2_credentials": [
                "The team demonstrates a strong academic background and relevant experience in protein design, which is crucial for the project's success.",
                "The team's extensive publication record and patent portfolio indicate a solid foundation in innovation and research capabilities.",
            ],
            "team_R6_complementarity": ["Diversity within the team may foster multidimensional development, particularly in addressing technical challenges."],
        },
        "feasibility": {
            "feasibility_R1_budget_detail": ["The structured budget plan indicates a thoughtful approach to financial management."],
            "feasibility_R3_infrastructure": ["Collaboration with the Institute of Biophysics may provide essential technical and market support."],
        },
    },
    "p4": {
        "strategy": {
            "strategy_R2_milestones": [
                "The milestone structure is compact and shows a unique approach compared to competitors, enhancing execution potential.",
                "The project has identified critical milestones for transitioning from foundational systems to commercial operations.",
            ],
            "strategy_R5_moat": ["Holding of the core patents enhances the project's competitive position in the market."],
            "strategy_R6_partners": ["Collaborations with top-tier hospitals in China offer critical clinical validation and market entry support."],
            "strategy_R7_team_fit": ["The leadership experience of Dr. Palti provides valuable commercial insights, enhancing the team's capability to navigate the commercialization process."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["The project objectives are clearly defined, focusing on the development of a second-generation tumor treatment solution."],
            "objectives_R7_problem_solution_fit": ["The project goals are well-aligned with addressing unmet needs in tumor treatment, indicating a clear market focus."],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": ["The second-generation TTF technology shows promise in terms of innovation and potential patient outcomes."],
            "advantages_R2_ip_status": ["Holding of the core patents enhances the project's competitive position in the market."],
            "advantages_R3_performance_proof": ["The technology claims significant improvements in patient survival rates and reduced harm, indicating strong innovation potential."],
            "advantages_R7_validation_signals": ["Collaborations with top-tier hospitals in China offer critical clinical validation and market entry support."],
        },
        "team": {
            "team_R2_credentials": ["The leadership experience of Dr. Palti provides valuable commercial insights, enhancing the team's capability to navigate the commercialization process."],
            "team_R6_complementarity": ["The team demonstrates a strong foundation with diverse expertise, particularly in medical device development, which is crucial for advancing the project."],
        },
        "feasibility": {
            "feasibility_R1_budget_detail": ["The budget allocation appears reasonable, with a significant portion earmarked for research and development."],
            "feasibility_R2_funding_secured": ["The angel financing target of 25 million yuan aligns with the overall budget allocation, indicating a structured financial approach."],
            "feasibility_R3_infrastructure": ["Collaborations with top-tier hospitals in China offer critical clinical validation and market entry support."],
            "feasibility_R5_risk_mitigation": ["The project claims zero legal risks, which could enhance investor if substantiated."],
        },
    },
    "p5": {
        "strategy": {
            "strategy_R2_milestones": [
                "Successful completion of initial milestones, such as flight validation, demonstrates progress.",
                "The project has established a timeline for key deliverables, indicating a structured approach.",
            ],
            "strategy_R5_moat": ["Innovations in propulsion technology, such as size reduction and cost efficiency, could provide a competitive edge."],
            "strategy_R6_partners": [
                "Collaboration with research institutions could enhance technological capabilities and market entry strategies.",
                "Partnerships with established organizations enhance technical capabilities and market competitiveness.",
            ],
            "strategy_R7_team_fit": ["Leadership experience in aerospace and technology development could guide effective decision-making."],
        },
        "objectives": {
            "objectives_R8_addressability": ["The goal to deliver 100 micro thrusters within five years is clearly defined and ambitious."],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": ["The proposed low dry weight thruster design aims to enhance efficiency and reduce costs, which is a positive aspect."],
            "advantages_R5_defensibility": ["Innovations in propulsion technology, such as size reduction and cost efficiency, could provide a competitive edge."],
            "advantages_R7_validation_signals": ["Successful completion of initial milestones, such as flight validation, demonstrates progress."],
        },
        "team": {
            "team_R2_credentials": ["Leadership experience in aerospace and technology development could guide effective decision-making."],
            "team_R6_complementarity": ["Initial team structure supports multi-functional operations, indicating a diverse skill set."],
        },
        "feasibility": {
            "feasibility_R1_budget_detail": ["Initial funding needs are clearly outlined, indicating a structured financial approach."],
            "feasibility_R2_funding_secured": ["The plan to secure funding through equity financing and local talent projects demonstrates a proactive approach."],
            "feasibility_R3_infrastructure": ["Partnerships with established organizations enhance technical capabilities and market competitiveness."],
            "feasibility_R4_technical_readiness": ["Successful completion of initial milestones, such as flight validation, demonstrates progress."],
            "feasibility_R8_sustainability": ["Projected positive cash flow from 2023 onwards indicates potential financial viability."],
        },
    },
    "p6": {
        "strategy": {
            "strategy_R6_partners": ["Collaboration with major companies like Alibaba and Tencent provides access to valuable resources and technical support."],
            "strategy_R7_team_fit": ["The team includes experienced leaders like Yang Guoqing, whose background in smart vehicle technology enhances the project's credibility."],
            "strategy_R8_timing": ["The primary objective of developing a high-performance computing platform is well-defined and aligns with industry trends."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["The focus on reducing application costs and enhancing operational value is relevant to the specialized operations sector."],
            "objectives_R5_why_now": ["The primary objective of developing a high-performance computing platform is well-defined and aligns with industry trends."],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": [
                "The SmartOSEK operating system employs a novel data-driven approach, distinguishing it from traditional programming methods.",
                "The project showcases promising technological innovations, particularly with the SmartOSEK operating system and the autonomous driving intelligent computing unit.",
            ],
        },
        "team": {
            "team_R2_credentials": [
                "The team includes experienced leaders like Yang Guoqing, whose background in smart vehicle technology enhances the project's credibility.",
                "The team demonstrates a strong foundation with relevant academic and industry experience, particularly in the field of autonomous driving technology.",
            ],
            "team_R6_complementarity": ["The diversity of the team, comprising experts from various academic institutions, may foster innovative solutions."],
        },
        "feasibility": {
            "feasibility_R2_funding_secured": [
                "The project has a structured financing plan that includes significant local funding, which can support product optimization and expansion.",
                "The financing plan includes substantial local funding, which can enhance project sustainability and reduce financial risks.",
            ],
            "feasibility_R3_infrastructure": ["Collaboration with major companies like Alibaba and Tencent provides access to valuable resources and technical support."],
            "feasibility_R5_risk_mitigation": ["Identifying market risks and having a risk management strategy in place indicates a proactive approach to potential challenges."],
        },
    },
    "p7": {
        "strategy": {
            "strategy_R2_milestones": ["The project outlines key stages for preclinical and clinical development."],
            "strategy_R5_moat": ["Identification of innovative technologies suggests a forward-thinking approach."],
            "strategy_R7_team_fit": ["Dr. Kong Na's leadership and publication history enhance the team's credibility."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["The project objectives with addressing significant patient needs in tumor immunotherapy, particularly in China."],
            "objectives_R4_unmet_need_evidence": ["The project targets a significant patient population in China, indicating a clear market need."],
            "objectives_R7_problem_solution_fit": ["Focus on innovative mRNA therapies suggests potential advancements in tumor immunotherapy."],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": [
                "The project presents innovative approaches in mRNA delivery systems, particularly with the linear mRNA scaffold and circularization technology.",
                "Circularization technology is expected to enhance translation efficiency and reduce immunogenicity.",
                "The linear mRNA scaffold template addresses specific challenges in mRNA delivery.",
            ],
            "advantages_R3_performance_proof": ["Preclinical safety tests suggest a favorable safety profile for the innovation."],
        },
        "team": {
            "team_R1_composition": ["The team exhibits strong academic credentials and leadership, particularly through Dr. Kong Na and Dr. Tao Wei, which enhances credibility in advancing mRNA drug delivery technologies."],
            "team_R2_credentials": ["Dr. Kong Na's leadership and publication history enhance the team's credibility."],
            "team_R5_capacity": ["The team’s investment of 3.15 million yuan enhances credibility."],
        },
        "feasibility": {
            "feasibility_R1_budget_detail": ["The budget of 35.2 million yuan indicates a significant financial commitment."],
            "feasibility_R2_funding_secured": ["The team’s investment of 3.15 million yuan enhances credibility."],
            "feasibility_R4_technical_readiness": ["Preclinical safety tests suggest a favorable safety profile for the innovation."],
        },
    },
    "p8": {
        "strategy": {
            "strategy_R5_moat": [
                "The emphasis on a unique AI-driven approach is intended to mitigate competitive pressures.",
                "The project outlines a strategy to leverage AI and partnerships to enhance market influence and address competitive pressures.",
            ],
            "strategy_R6_partners": [
                "Leveraging existing partnerships with Merck and Fujifilm may enhance credibility and resource access.",
                "Collaboration with strategic partners like Merck and Fujifilm may provide additional resources and expertise.",
            ],
            "strategy_R7_team_fit": ["The management team structure supports the project's innovative objectives."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["The project targets a critical need for antiviral therapies in high-risk populations."],
            "objectives_R5_why_now": ["The project's objectives with a pressing public health need for antiviral therapies, particularly targeting high-risk populations."],
            "objectives_R6_competitive_context": ["The emphasis on a unique AI-driven approach is intended to mitigate competitive pressures."],
            "objectives_R7_problem_solution_fit": [
                "Focus on lung-targeted delivery is relevant given the respiratory nature of many viral infections.",
                "Focus on stability and controlled release addresses common challenges in drug delivery systems.",
            ],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": [
                "The project showcases innovative approaches in drug delivery through the development of lung-targeted stable lipid nanoparticles and the integration of AI models.",
                "Utilization of advanced gene engineering techniques for improved specificity in drug delivery.",
                "Integration of AI models is a distinguishing feature that enhances precision.",
            ],
            "advantages_R5_defensibility": ["The emphasis on a unique AI-driven approach is intended to mitigate competitive pressures."],
        },
        "team": {
            "team_R1_composition": ["The management team structure supports the project's innovative objectives."],
            "team_R6_complementarity": [
                "Diverse expertise within the team enhances capability to address complex challenges.",
                "The project team demonstrates a structured leadership approach with a diverse set of experts, which is essential for tackling the complexities of drug development.",
            ],
        },
        "feasibility": {
            "feasibility_R2_funding_secured": ["Phased funding strategy may assist in attracting strategic investors."],
            "feasibility_R3_infrastructure": ["Utilization of existing laboratory resources may help reduce initial investment pressure."],
            "feasibility_R5_risk_mitigation": [
                "Identified technical mitigation strategies for RNA drug delivery challenges through in vivo testing.",
                "Recognition of funding pressure as a significant risk demonstrates awareness of financial sustainability.",
            ],
        },
    },
    "pA": {
        "strategy": {
            "strategy_R5_moat": ["The new IGBT technology platform claims industry-leading performance metrics, which could provide a competitive edge."],
            "strategy_R6_partners": [
                "The project strategy includes forming joint ventures to enhance market presence and optimize revenue.",
                "The strategy of forming joint ventures with established semiconductor companies can enhance market credibility and access to top-tier clients.",
            ],
            "strategy_R7_team_fit": ["The core team has over ten years of experience in leading semiconductor companies, providing a strong foundation for project execution."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["The project has clear objectives aimed at enhancing semiconductor production capabilities in response to domestic demand."],
            "objectives_R7_problem_solution_fit": ["The project's objectives are focused on establishing a comprehensive analog semiconductor industrial chain in China, aligning with market needs."],
        },
        "advantages": {
            "advantages_R2_ip_status": ["The organization has applied for numerous patents, indicating a proactive approach to intellectual property and innovation."],
            "advantages_R3_performance_proof": ["The new IGBT technology platform claims industry-leading performance metrics, which could provide a competitive edge."],
        },
        "team": {
            "team_R2_credentials": ["The team's diverse academic backgrounds, including multiple PhDs and Master's degrees, enhance its technical capabilities."],
            "team_R3_track_record": [
                "The core team has over ten years of experience in leading semiconductor companies, providing a strong foundation for project execution.",
                "The team possesses significant experience in the semiconductor industry, which is crucial for project execution.",
            ],
            "team_R4_governance": ["Organizational structure supports market and customer service needs, facilitating effective execution of the strategy."],
        },
        "feasibility": {
            "feasibility_R2_funding_secured": ["The planned investment of 260 billion Yuan indicates strong financial backing and commitment to the project's objectives."],
            "feasibility_R3_infrastructure": ["Collaborations with domestic universities and research institutions can foster innovation and provide access to new technologies."],
        },
    },
    "pB": {
        "strategy": {
            "strategy_R1_gtm": ["The company has established relationships with over 20 world-class customers, indicating market validation and potential for growth."],
            "strategy_R5_moat": ["The strategy to leverage advanced battery technology positions the company well in the competitive landscape."],
            "strategy_R6_partners": ["The implementation strategy for 大瞬科技 appears promising, leveraging existing partnerships and market opportunities."],
            "strategy_R7_team_fit": ["The leadership team possesses extensive engineering experience, which is crucial for managing production processes effectively."],
            "strategy_R8_timing": ["The urgency of the need for technological independence is well articulated, reflecting a significant market opportunity."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["大瞬科技's objectives well with the urgent need for technological independence among Chinese smartphone manufacturers."],
            "objectives_R3_buyer_pathway": ["The company has established relationships with over 20 world-class customers, indicating market validation and potential for growth."],
            "objectives_R5_why_now": ["The urgency of the need for technological independence is well articulated, reflecting a significant market opportunity."],
            "objectives_R6_competitive_context": ["The strategy to leverage advanced battery technology positions the company well in the competitive landscape."],
            "objectives_R7_problem_solution_fit": ["The objectives are closely aligned with the market demand for high-end fast charging chips and power management solutions."],
        },
        "advantages": {
            "advantages_R2_ip_status": [
                "The project holds multiple patents, indicating a strong foundation for technological innovation in power management and Bluetooth connectivity.",
                "大瞬科技 demonstrates strong potential for innovation with proprietary technologies and patents.",
            ],
            "advantages_R3_performance_proof": ["The low defect rate (DPPM below 5) suggests a high level of quality control and production efficiency."],
            "advantages_R6_platform_potential": ["The diverse product portfolio positions the company well to capitalize on market opportunities in fast charging and power management."],
            "advantages_R7_validation_signals": ["The company has established relationships with over 20 world-class customers, indicating market validation and potential for growth."],
        },
        "team": {
            "team_R2_credentials": [
                "The leadership team possesses extensive engineering experience, which is crucial for managing production processes effectively.",
                "The team at 大瞬科技 shows potential with a strong engineering background and a history of successful project delivery.",
            ],
            "team_R3_track_record": ["The team has a proven track record of delivering over 3 billion units without production interruptions, indicating strong operational capabilities."],
        },
        "feasibility": {
            "feasibility_R4_technical_readiness": ["The low defect rate (DPPM below 5) suggests a high level of quality control and production efficiency."],
            "feasibility_R6_resource_timeline_fit": ["The company has a strong track record of timely deliveries, indicating effective operational execution."],
            "feasibility_R8_sustainability": [
                "The feasibility of 大瞬科技's project is supported by a diverse product portfolio and a history of successful deliveries.",
                "The diverse product portfolio positions the company well to capitalize on market opportunities in fast charging and power management.",
            ],
        },
    },
    "pC": {
        "strategy": {
            "strategy_R2_milestones": ["The project aims to establish clear milestones for expansion, which is critical for tracking progress."],
            "strategy_R6_partners": ["The initiative emphasizes the importance of partnerships with schools and enterprises to integrate industry standards."],
            "strategy_R7_team_fit": ["Team members possess operational awareness and project management capabilities, aligning with project goals."],
        },
        "objectives": {
            "objectives_R1_problem_clarity": ["The initiative reflects responsiveness to the demand for applied talent cultivation."],
            "objectives_R8_addressability": [
                "The overall goal of establishing 100 colleges is clearly defined and focused.",
                "The project has a clear primary objective of establishing 100 colleges by 2024, which aligns with industry needs.",
            ],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": ["Standardized educational products are designed to meet specific industry needs, potentially offering targeted solutions."],
        },
        "team": {
            "team_R2_credentials": ["Team members possess operational awareness and project management capabilities, aligning with project goals."],
            "team_R3_track_record": ["Experience in collaborating with government enhances credibility and may provide additional resources."],
            "team_R4_governance": [
                "The leadership structure is clear, with a General Manager responsible for operational oversight, which can enhance project efficiency.",
                "The team demonstrates a structured leadership approach with a General Manager overseeing operations, which can enhance accountability and execution.",
            ],
        },
        "feasibility": {
            "feasibility_R3_infrastructure": [
                "Investment in servers and workspace provides necessary infrastructure support for long-term operations.",
                "The project has a solid foundation with infrastructure investments and a low financial risk strategy.",
            ],
            "feasibility_R4_technical_readiness": ["Strong operational capabilities and resource base indicate readiness for implementation."],
            "feasibility_R5_risk_mitigation": ["Low financial risk due to contingent resource investments enhances project viability."],
        },
    },
    "pD": {
        "strategy": {
            "strategy_R1_gtm": ["The reliance on distributors and agents may facilitate rapid market entry and leverage existing distribution networks."],
            "strategy_R2_milestones": ["The project has a clear timeline for going public and focuses on developing high-efficiency charge pump chips, indicating a strong market orientation."],
            "strategy_R5_moat": ["High efficiency claims for the charge pump (-) suggest a competitive edge, pending further validation against industry standards."],
            "strategy_R7_team_fit": ["The team includes a substantial number of engineers (over ), enhancing R&D capabilities and responsiveness to market needs."],
        },
        "objectives": {
            "objectives_R6_competitive_context": ["High efficiency claims for the charge pump (-) suggest a competitive edge, pending further validation against industry standards."],
        },
        "advantages": {
            "advantages_R1_mechanism_novelty": ["The project showcases promising innovations, particularly in solar charging technology and charge pump efficiency."],
            "advantages_R3_performance_proof": ["The solar charging chip demonstrates potential for superior performance in challenging conditions, enhancing market differentiation."],
            "advantages_R5_defensibility": ["The implementation of a knowledge protection strategy aims to mitigate supply risks associated with insufficient domestic intellectual property protection."],
        },
        "team": {
            "team_R1_composition": ["The team includes a substantial number of engineers (over ), enhancing R&D capabilities and responsiveness to market needs."],
            "team_R2_credentials": ["Zhang Liye's legal background in microelectronics and biotechnology could provide valuable guidance on compliance and intellectual property management."],
            "team_R4_governance": ["The stable management structure with employee ownership may foster commitment and improve decision-making efficiency."],
            "team_R6_complementarity": ["The team demonstrates a strong technical background, particularly with a high proportion of engineers, which supports R&D capabilities."],
        },
        "feasibility": {
            "feasibility_R3_infrastructure": ["The establishment of a supply chain procurement system supports operational efficiency and cost reduction."],
            "feasibility_R5_risk_mitigation": ["The implementation of a knowledge protection strategy aims to mitigate supply risks associated with insufficient domestic intellectual property protection."],
            "feasibility_R6_resource_timeline_fit": ["The company is actively working to expand production capacity to meet market demand, indicating responsiveness to growth opportunities."],
        },
    },
}
