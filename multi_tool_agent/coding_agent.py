import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from pydantic import BaseModel, Field
from google.adk.models.lite_llm import LiteLlm

# --- Constants ---
APP_NAME = "startup_website_app"
USER_ID = "startup_user"
SESSION_ID = "website_session"
MODEL_OLLAMA = "openai/qwen3:latest"  # Replace with your preferred model

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Coding Agent ---
class StartupCodingAgent(BaseAgent):
    """
    Custom agent for generating a startup website based on a startup concept.
    
    This agent orchestrates a workflow that:
    1. Analyzes the startup concept
    2. Creates website content structure
    3. Generates HTML/CSS for the landing page
    4. Creates necessary JavaScript functionality
    5. Assembles the complete website package
    """

    # --- Field Declarations for Pydantic ---
    concept_analyzer: LlmAgent
    content_creator: LlmAgent
    html_generator: LlmAgent
    css_generator: LlmAgent
    js_generator: LlmAgent
    website_assembler: LlmAgent

    # Setup composite agents
    frontend_generator: SequentialAgent

    # model_config allows setting Pydantic configurations
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        concept_analyzer: LlmAgent,
        content_creator: LlmAgent,
        html_generator: LlmAgent,
        css_generator: LlmAgent,
        js_generator: LlmAgent,
        website_assembler: LlmAgent,
    ):
        """
        Initializes the StartupCodingAgent.

        Args:
            name: The name of the agent.
            concept_analyzer: LlmAgent to analyze the startup concept.
            content_creator: LlmAgent to create website content structure.
            html_generator: LlmAgent to generate HTML code.
            css_generator: LlmAgent to generate CSS code.
            js_generator: LlmAgent to generate JavaScript code.
            website_assembler: LlmAgent to assemble the complete website.
        """
        # Create internal composite agents
        frontend_generator = SequentialAgent(
            name="FrontendGenerator", 
            sub_agents=[html_generator, css_generator, js_generator]
        )

        # Define the sub_agents list
        sub_agents_list = [
            concept_analyzer,
            content_creator,
            frontend_generator,
            website_assembler,
        ]

        # Initialize using parent class constructor
        super().__init__(
            name=name,
            concept_analyzer=concept_analyzer,
            content_creator=content_creator,
            html_generator=html_generator,
            css_generator=css_generator,
            js_generator=js_generator,
            website_assembler=website_assembler,
            frontend_generator=frontend_generator,
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the website generation workflow.
        """
        logger.info(f"[{self.name}] Starting website generation workflow.")

        # 1. Analyze the startup concept
        logger.info(f"[{self.name}] Running ConceptAnalyzer...")
        async for event in self.concept_analyzer.run_async(ctx):
            logger.info(f"[{self.name}] Event from ConceptAnalyzer: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # Check if concept was analyzed successfully
        if "website_requirements" not in ctx.session.state or not ctx.session.state["website_requirements"]:
            logger.error(f"[{self.name}] Failed to analyze concept. Aborting workflow.")
            return

        logger.info(f"[{self.name}] Website requirements defined: {ctx.session.state.get('website_requirements')}")

        # 2. Create website content structure
        logger.info(f"[{self.name}] Running ContentCreator...")
        async for event in self.content_creator.run_async(ctx):
            logger.info(f"[{self.name}] Event from ContentCreator: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info(f"[{self.name}] Website content structure created: {ctx.session.state.get('website_content')}")

        # 3. Generate frontend code (HTML, CSS, JS) sequentially
        logger.info(f"[{self.name}] Running FrontendGenerator...")
        async for event in self.frontend_generator.run_async(ctx):
            logger.info(f"[{self.name}] Event from FrontendGenerator: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info(f"[{self.name}] Frontend components generated.")

        # 4. Assemble the complete website
        logger.info(f"[{self.name}] Running WebsiteAssembler...")
        async for event in self.website_assembler.run_async(ctx):
            logger.info(f"[{self.name}] Event from WebsiteAssembler: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info(f"[{self.name}] Website generation workflow completed.")

# --- Define the individual LLM agents ---
concept_analyzer = LlmAgent(
    name="ConceptAnalyzer",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a startup concept analyzer. Analyze the startup concept in session state with key 
    'startup_concept' and extract the key information needed for website development.
    
    Identify:
    1. Company name and tagline
    2. Value proposition
    3. Target audience
    4. Key features or benefits (3-5)
    5. Visual style guidance (colors, imagery themes)
    6. Call-to-action strategies
    
    Format your response as a structured JSON-like object with these components.
    """,
    input_schema=None,
    output_key="website_requirements",
)

content_creator = LlmAgent(
    name="ContentCreator",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a website content creator. Based on the website requirements in session state with key
    'website_requirements', create the content structure for a startup landing page.
    
    Generate:
    1. Hero section headline and subheadline
    2. About section content (2-3 paragraphs)
    3. Features section with brief descriptions
    4. Testimonial placeholders (2-3)
    5. Call-to-action button text and surrounding content
    6. Contact section text
    
    Format as a structured content plan that can be used by the HTML generator.
    """,
    input_schema=None,
    output_key="website_content",
)

html_generator = LlmAgent(
    name="HtmlGenerator",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are an HTML developer. Create the HTML structure for a startup landing page based on
    the website requirements in session state with key 'website_requirements' and the content in 
    'website_content'.
    
    Create semantic HTML5 with:
    1. Proper document structure (doctype, head, body)
    2. Header with navigation
    3. Hero section
    4. About section
    5. Features section with cards or suitable components
    6. Testimonials section
    7. Call-to-action section
    8. Footer with contact info
    
    Use appropriate classes for styling with CSS. Include comments to explain the structure.
    Output only the complete HTML code, nothing else.
    """,
    input_schema=None,
    output_key="html_code",
)

css_generator = LlmAgent(
    name="CssGenerator",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a CSS developer. Create the styling for the startup landing page based on
    the HTML in session state with key 'html_code' and the visual guidance in 'website_requirements'.
    
    Create responsive CSS with:
    1. Modern, clean styling
    2. Responsive design (mobile-first approach)
    3. Animations for key interactions
    4. Consistent color scheme based on the requirements
    5. Typography that enhances readability
    
    Include media queries for different screen sizes. Add comments to explain styling decisions.
    Output only the complete CSS code, nothing else.
    """,
    input_schema=None,
    output_key="css_code",
)

js_generator = LlmAgent(
    name="JsGenerator",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a JavaScript developer. Create the interactive functionality for the startup landing page
    based on the HTML in session state with key 'html_code'.
    
    Create JavaScript that handles:
    1. Smooth scrolling for navigation
    2. Simple form validation for contact or signup forms
    3. Mobile menu toggle
    4. Any animations or interactions that enhance UX
    
    Use vanilla JavaScript (no frameworks). Include comments to explain the code.
    Output only the complete JavaScript code, nothing else.
    """,
    input_schema=None,
    output_key="js_code",
)

website_assembler = LlmAgent(
    name="WebsiteAssembler",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a website assembler. Create a complete, deployable website by combining:
    1. HTML from session state with key 'html_code'
    2. CSS from session state with key 'css_code'
    3. JavaScript from session state with key 'js_code'
    
    Your tasks:
    1. Ensure all components work together
    2. Create a file structure description
    3. List any additional assets needed (images, fonts)
    4. Provide instructions for local testing
    5. Package everything into a format ready for deployment
    
    Generate a deployment-ready package with clear documentation.
    """,
    input_schema=None,
    output_key="website_package",
)

# --- Create the custom agent instance ---
startup_coding_agent = StartupCodingAgent(
    name="StartupCodingAgent",
    concept_analyzer=concept_analyzer,
    content_creator=content_creator,
    html_generator=html_generator,
    css_generator=css_generator,
    js_generator=js_generator,
    website_assembler=website_assembler,
)

# --- Setup Runner and Session ---
session_service = InMemorySessionService()
initial_state = {}  # Empty initial state, will be populated with startup concept
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initial_state
)

runner = Runner(
    agent=startup_coding_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# --- Function to Interact with the Agent ---
def call_coding_agent(startup_concept: str):
    """
    Sends a startup concept to the coding agent and runs the website generation workflow.
    
    Args:
        startup_concept: The final startup concept summary from the ideation agent
    """
    current_session = session_service.get_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )
    
    if not current_session:
        logger.error("Session not found!")
        return

    # Update the startup concept in the session
    current_session.state["startup_concept"] = startup_concept
    logger.info(f"Updated session with startup concept.")

    # Create message content
    content = types.Content(
        role='user', 
        parts=[types.Part(text=f"Generate a website for this startup concept: {startup_concept[:100]}...")]
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
    print("\n--- Website Generation Result ---")
    print("Final Response: ", final_response)

    # Get final session state
    final_session = session_service.get_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )
    
    print("\nWebsite Package Created:")
    # In a real implementation, you might save files to disk here
    
    # Return the website package for potential use by deployment agent
    return final_session.state.get("website_package")

# --- Example of connecting Ideation and Coding Agents ---
def integrated_workflow(domain: str):
    """
    Demonstrates the complete workflow from ideation to coding.
    """
    # First, run the ideation agent
    from startup_ideation_agent import call_ideation_agent
    
    print(f"Starting ideation process for domain: {domain}")
    startup_concept = call_ideation_agent(domain)
    
    # Then, pass the result to the coding agent
    print("Generating website based on the startup concept...")
    website_package = call_coding_agent(startup_concept)
    
    # This could then be passed to a deployment agent
    print("Website generated. Ready for deployment.")
    return website_package

# --- Example usage ---
if __name__ == "__main__":
    # This would normally get the concept from the ideation agent
    example_concept = """
    EcoNest - "Sustainable living, simplified."
    
    Value Proposition: EcoNest provides apartment dwellers with space-efficient, sustainable living 
    products that reduce environmental impact while enhancing urban living spaces.
    
    Target Market: Environmentally conscious urban professionals ages 25-40 living in apartments in 
    major cities, who value sustainability but have limited space.
    
    Business Model: Direct-to-consumer subscription model with modular product ecosystem and 
    carbon offset program.
    
    Competitive Advantage: Unique modular design system that allows products to adapt to changing 
    needs and small spaces, with comprehensive sustainability metrics for each product.
    """
    
    # Run the coding agent with the example concept
    website_package = call_coding_agent(example_concept)