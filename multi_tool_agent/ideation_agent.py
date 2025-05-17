import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from pydantic import BaseModel, Field
from google.adk.models.lite_llm import LiteLlm

# --- Constants ---
APP_NAME = "startup_ideation_app"
USER_ID = "startup_user"
SESSION_ID = "ideation_session"
MODEL_OLLAMA = "openai/qwen3:latest"  # You can replace with your preferred model

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Orchestrator Agent ---
class StartupIdeationAgent(BaseAgent):
    """
    Custom agent for startup idea generation, market research, and refinement.
    
    This agent orchestrates a workflow that:
    1. Generates initial startup ideas based on user input
    2. Conducts market research on those ideas
    3. Critiques the ideas based on market research
    4. Refines the ideas based on critique
    5. Evaluates feasibility
    6. Summarizes the final startup concept
    """

    # --- Field Declarations for Pydantic ---
    idea_generator: LlmAgent
    market_researcher: LlmAgent
    idea_critic: LlmAgent
    idea_refiner: LlmAgent
    feasibility_checker: LlmAgent
    final_summarizer: LlmAgent

    # Setup agent composition
    refine_loop_agent: LoopAgent
    evaluation_agent: SequentialAgent

    # model_config allows setting Pydantic configurations
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        idea_generator: LlmAgent,
        market_researcher: LlmAgent,
        idea_critic: LlmAgent,
        idea_refiner: LlmAgent,
        feasibility_checker: LlmAgent,
        final_summarizer: LlmAgent,
    ):
        """
        Initializes the StartupIdeationAgent.

        Args:
            name: The name of the agent.
            idea_generator: LlmAgent to generate initial startup ideas.
            market_researcher: LlmAgent to research market potential.
            idea_critic: LlmAgent to critique ideas based on research.
            idea_refiner: LlmAgent to refine ideas based on criticism.
            feasibility_checker: LlmAgent to evaluate the feasibility.
            final_summarizer: LlmAgent to provide final startup concept.
        """
        # Create internal composite agents
        refine_loop_agent = LoopAgent(
            name="CritiqueRefineLoop", 
            sub_agents=[idea_critic, idea_refiner], 
            max_iterations=3
        )
        
        evaluation_agent = SequentialAgent(
            name="FinalEvaluation", 
            sub_agents=[feasibility_checker, final_summarizer]
        )

        # Define the sub_agents list
        sub_agents_list = [
            idea_generator,
            market_researcher,
            refine_loop_agent,
            evaluation_agent,
        ]

        # Initialize using parent class constructor
        super().__init__(
            name=name,
            idea_generator=idea_generator,
            market_researcher=market_researcher,
            idea_critic=idea_critic,
            idea_refiner=idea_refiner,
            feasibility_checker=feasibility_checker,
            final_summarizer=final_summarizer,
            refine_loop_agent=refine_loop_agent,
            evaluation_agent=evaluation_agent,
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the startup ideation workflow.
        """
        logger.info(f"[{self.name}] Starting startup ideation workflow.")

        # 1. Initial Idea Generation
        logger.info(f"[{self.name}] Running IdeaGenerator...")
        async for event in self.idea_generator.run_async(ctx):
            logger.info(f"[{self.name}] Event from IdeaGenerator: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # Check if ideas were generated before proceeding
        if "startup_idea" not in ctx.session.state or not ctx.session.state["startup_idea"]:
            logger.error(f"[{self.name}] Failed to generate initial ideas. Aborting workflow.")
            return  # Stop processing if initial ideas failed

        logger.info(f"[{self.name}] Ideas generated: {ctx.session.state.get('startup_idea')}")

        # 2. Market Research
        logger.info(f"[{self.name}] Running MarketResearcher...")
        async for event in self.market_researcher.run_async(ctx):
            logger.info(f"[{self.name}] Event from MarketResearcher: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info(f"[{self.name}] Market research completed: {ctx.session.state.get('market_research')}")

        # 3. Critique and Refinement Loop
        logger.info(f"[{self.name}] Running CritiqueRefineLoop...")
        async for event in self.refine_loop_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from CritiqueRefineLoop: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info(f"[{self.name}] Idea after refinement: {ctx.session.state.get('startup_idea')}")

        # 4. Final Evaluation (Feasibility Check and Summary)
        logger.info(f"[{self.name}] Running FinalEvaluation...")
        async for event in self.evaluation_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from FinalEvaluation: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # 5. Check feasibility score for conditional logic
        feasibility_score = ctx.session.state.get("feasibility_score", 0)
        logger.info(f"[{self.name}] Feasibility score: {feasibility_score}")

        if feasibility_score < 5:
            logger.info(f"[{self.name}] Low feasibility. Requesting idea pivoting...")
            # Store that we need to pivot
            ctx.session.state["needs_pivot"] = True
            # Set a pivot prompt in the state
            ctx.session.state["pivot_request"] = "The original idea has low feasibility. Please pivot to an adjacent market or modify the core concept while preserving the main value proposition."
            # Run the idea refiner one more time with pivot guidance
            async for event in self.idea_refiner.run_async(ctx):
                logger.info(f"[{self.name}] Event from IdeaRefiner (Pivot): {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
            
            # Run final summarizer one more time with the pivoted idea
            async for event in self.final_summarizer.run_async(ctx):
                logger.info(f"[{self.name}] Event from FinalSummarizer (After Pivot): {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
        else:
            logger.info(f"[{self.name}] Feasibility is acceptable. Proceeding with current idea.")
            ctx.session.state["needs_pivot"] = False

        logger.info(f"[{self.name}] Ideation workflow completed.")

# --- Define the individual LLM agents ---
idea_generator = LlmAgent(
    name="IdeaGenerator",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a creative startup idea generator. Generate 3 potential startup ideas based on the 
    industry, problem space, or technology provided in session state with key 'domain'.
    For each idea include:
    1. A catchy name
    2. A one-sentence description
    3. The core value proposition
    4. The target audience
    
    Format your response as a list of ideas separated by '---'.
    """,
    input_schema=None,
    output_key="startup_idea",  # Key for storing output in session state
)

market_researcher = LlmAgent(
    name="MarketResearcher",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a market research analyst. Analyze the market potential for the startup ideas provided 
    in session state with key 'startup_idea'. 
    Research and report on:
    1. Current market size and growth rate
    2. Key competitors and similar solutions
    3. Unique market opportunities
    4. Potential challenges or barriers to entry
    
    Be realistic and data-driven in your assessment based on your knowledge of current market trends.
    """,
    input_schema=None,
    output_key="market_research",  # Key for storing market research in session state
)

idea_critic = LlmAgent(
    name="IdeaCritic",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a startup idea critic. Review the startup idea in session state with key 'startup_idea'
    and the market research in session state with key 'market_research'.
    
    Provide constructive criticism on:
    1. Market fit
    2. Value proposition clarity
    3. Differentiation from competitors
    4. Potential business model viability
    
    Be honest but constructive. Focus on how the idea could be improved.
    """,
    input_schema=None,
    output_key="idea_criticism",  # Key for storing criticism in session state
)

idea_refiner = LlmAgent(
    name="IdeaRefiner",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a startup idea refiner. Refine the startup idea in session state with key 'startup_idea'
    based on the criticism in session state with key 'idea_criticism' and market research in 'market_research'.
    
    If 'needs_pivot' is true and 'pivot_request' exists in the session state, follow those pivot instructions.
    
    Improve the idea by:
    1. Clarifying the value proposition
    2. Sharpening the target audience
    3. Addressing any market fit issues
    4. Enhancing differentiation from competitors
    
    Output the refined idea with the same structure as the original, but with improvements.
    """,
    input_schema=None,
    output_key="startup_idea",  # Overwrites the original idea
)

feasibility_checker = LlmAgent(
    name="FeasibilityChecker",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a startup feasibility analyst. Evaluate the feasibility of the startup idea
    in session state with key 'startup_idea' based on:
    1. Technical feasibility
    2. Market readiness
    3. Resource requirements (funding, talent, etc.)
    4. Regulatory considerations
    5. Time to market
    
    Rate the overall feasibility on a scale of 1-10 (10 being highest feasibility).
    Provide your numerical score, followed by a brief explanation.
    """,
    input_schema=None,
    output_key="feasibility_analysis",  # Stores the analysis
)

final_summarizer = LlmAgent(
    name="FinalSummarizer",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a startup concept summarizer. Create a comprehensive summary of the final
    startup idea from session state with key 'startup_idea', incorporating insights from:
    - 'market_research'
    - 'feasibility_analysis'
    
    Your summary should include:
    1. The final startup name and tagline
    2. Clear value proposition
    3. Target market and user personas
    4. Business model overview
    5. Competitive advantage
    6. Next steps for validation
    
    Create a concise, compelling summary that could be presented to potential investors.
    
    Also, extract the feasibility score (1-10) from the feasibility analysis and store only
    that numeric value in your response.
    """,
    input_schema=None,
    output_key="final_concept_summary",  # Final output of the workflow
)

# --- Create the custom agent instance ---
startup_ideation_agent = StartupIdeationAgent(
    name="StartupIdeationAgent",
    idea_generator=idea_generator,
    market_researcher=market_researcher,
    idea_critic=idea_critic,
    idea_refiner=idea_refiner,
    feasibility_checker=feasibility_checker,
    final_summarizer=final_summarizer,
)

# --- Setup Runner and Session ---
session_service = InMemorySessionService()
initial_state = {"domain": "sustainable living products for urban apartments"}
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initial_state  # Pass initial state here
)
logger.info(f"Initial session state: {session.state}")

runner = Runner(
    agent=startup_ideation_agent,  # Pass the custom orchestrator agent
    app_name=APP_NAME,
    session_service=session_service
)

# --- Function to Interact with the Agent ---
def call_ideation_agent(domain: str):
    """
    Sends a new domain/industry focus to the agent and runs the ideation workflow.
    
    Args:
        domain: The industry, problem space, or technology focus area
    """
    current_session = session_service.get_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )
    
    if not current_session:
        logger.error("Session not found!")
        return

    # Update the domain in the session
    current_session.state["domain"] = domain
    logger.info(f"Updated session state domain to: {domain}")

    # Create message content
    content = types.Content(
        role='user', 
        parts=[types.Part(text=f"Generate startup ideas in the domain of: {domain}")]
    )
    
    # Run the agent
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    # Collect final response
    final_response = "No final response captured."
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            logger.info(f"Potential final response from [{event.author}]: {event.content.parts[0].text}")
            final_response = event.content.parts[0].text

    # Print results
    print("\n--- Startup Ideation Result ---")
    print("Final Response: ", final_response)

    # Get final session state
    final_session = session_service.get_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )
    
    print("\nFinal Session State:")
    import json
    print(json.dumps(final_session.state, indent=2))
    print("-------------------------------\n")
    
    # Return the final concept for potential use by other agents
    return final_session.state.get("final_concept_summary")

# --- Example usage ---
if __name__ == "__main__":
    # Example domains to ideate on
    example_domains = [
        "health tech for seniors",
        "sustainable food delivery",
        "remote work productivity tools",
        "educational technology for middle schools"
    ]
    
    # Run the agent with a selected domain
    selected_domain = example_domains[1]  # Change index to test different domains
    result = call_ideation_agent(selected_domain)