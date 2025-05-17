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
APP_NAME = "startup_deployment_app"
USER_ID = "startup_user"
SESSION_ID = "deployment_session"
MODEL_OLLAMA = "openai/qwen3:latest"  # Replace with your preferred model

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Deployment Agent ---
class StartupDeploymentAgent(BaseAgent):
    """
    Custom agent for deploying a startup website.
    
    This agent orchestrates a workflow that:
    1. Analyzes the website package
    2. Prepares files for deployment
    3. Configures deployment settings
    4. Executes deployment
    5. Performs post-deployment verification
    """

    # --- Field Declarations for Pydantic ---
    package_analyzer: LlmAgent
    deployment_preparer: LlmAgent
    infrastructure_configurator: LlmAgent
    deployer: LlmAgent
    verifier: LlmAgent

    # Setup composite agents
    deployment_sequence: SequentialAgent

    # model_config allows setting Pydantic configurations
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        package_analyzer: LlmAgent,
        deployment_preparer: LlmAgent,
        infrastructure_configurator: LlmAgent,
        deployer: LlmAgent,
        verifier: LlmAgent,
    ):
        """
        Initializes the StartupDeploymentAgent.

        Args:
            name: The name of the agent.
            package_analyzer: LlmAgent to analyze the website package.
            deployment_preparer: LlmAgent to prepare files for deployment.
            infrastructure_configurator: LlmAgent to configure deployment infrastructure.
            deployer: LlmAgent to execute the deployment.
            verifier: LlmAgent to verify successful deployment.
        """
        # Create internal composite agents
        deployment_sequence = SequentialAgent(
            name="DeploymentSequence", 
            sub_agents=[deployment_preparer, infrastructure_configurator, deployer]
        )

        # Define the sub_agents list
        sub_agents_list = [
            package_analyzer,
            deployment_sequence,
            verifier,
        ]

        # Initialize using parent class constructor
        super().__init__(
            name=name,
            package_analyzer=package_analyzer,
            deployment_preparer=deployment_preparer,
            infrastructure_configurator=infrastructure_configurator,
            deployer=deployer,
            verifier=verifier,
            deployment_sequence=deployment_sequence,
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the deployment workflow.
        """
        logger.info(f"[{self.name}] Starting website deployment workflow.")

        # 1. Analyze the website package
        logger.info(f"[{self.name}] Running PackageAnalyzer...")
        async for event in self.package_analyzer.run_async(ctx):
            logger.info(f"[{self.name}] Event from PackageAnalyzer: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # Check if package was analyzed successfully
        if "deployment_plan" not in ctx.session.state or not ctx.session.state["deployment_plan"]:
            logger.error(f"[{self.name}] Failed to analyze package. Aborting workflow.")
            return

        logger.info(f"[{self.name}] Deployment plan created: {ctx.session.state.get('deployment_plan')}")

        # 2. Run the deployment sequence
        logger.info(f"[{self.name}] Running DeploymentSequence...")
        async for event in self.deployment_sequence.run_async(ctx):
            logger.info(f"[{self.name}] Event from DeploymentSequence: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info(f"[{self.name}] Deployment executed.")

        # 3. Verify deployment
        logger.info(f"[{self.name}] Running Verifier...")
        async for event in self.verifier.run_async(ctx):
            logger.info(f"[{self.name}] Event from Verifier: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # Check verification result
        verification_result = ctx.session.state.get("verification_result", {})
        if verification_result.get("success", False):
            logger.info(f"[{self.name}] Deployment verified successfully.")
        else:
            logger.error(f"[{self.name}] Deployment verification failed: {verification_result.get('issues', [])}")
            
            # Optional: Implement automatic fixes for common issues
            if "needs_fixes" in verification_result and verification_result["needs_fixes"]:
                logger.info(f"[{self.name}] Attempting automatic fixes...")
                # Implementation for automatic fixes would go here
                pass

        logger.info(f"[{self.name}] Deployment workflow completed.")

# --- Define the individual LLM agents ---
package_analyzer = LlmAgent(
    name="PackageAnalyzer",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a deployment analyst. Analyze the website package in session state with key 
    'website_package' and create a deployment plan.
    
    Your analysis should include:
    1. File structure assessment
    2. Deployment requirements
    3. Recommended hosting platform (static hosting, serverless, etc.)
    4. Configuration needs
    5. Potential deployment challenges
    
    Create a structured deployment plan that can guide the rest of the deployment process.
    """,
    input_schema=None,
    output_key="deployment_plan",
)

deployment_preparer = LlmAgent(
    name="DeploymentPreparer",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a deployment preparation specialist. Based on the deployment plan in session state with key
    'deployment_plan' and the website package in 'website_package', prepare the files for deployment.
    
    Your tasks:
    1. Organize files according to hosting requirements
    2. Create any necessary configuration files (e.g., .htaccess, netlify.toml, etc.)
    3. Optimize assets for deployment if needed
    4. Prepare environment-specific configurations
    5. Generate deployment scripts if necessary
    
    Output a structured description of the prepared deployment assets.
    """,
    input_schema=None,
    output_key="prepared_assets",
)

infrastructure_configurator = LlmAgent(
    name="InfrastructureConfigurator",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are an infrastructure configuration specialist. Based on the deployment plan in session state
    with key 'deployment_plan' and prepared assets in 'prepared_assets', configure the necessary infrastructure.
    
    Your tasks:
    1. Define hosting environment settings
    2. Configure domain and DNS settings
    3. Set up CDN if applicable
    4. Configure security settings
    5. Prepare monitoring or analytics
    
    Output a structured configuration that can be used for deployment execution.
    """,
    input_schema=None,
    output_key="infrastructure_config",
)

deployer = LlmAgent(
    name="Deployer",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a deployment executor. Based on the prepared assets in session state with key 'prepared_assets'
    and infrastructure configuration in 'infrastructure_config', execute the deployment process.
    
    Your tasks:
    1. Generate deployment commands for the specified hosting platform
    2. Execute file transfers or uploads
    3. Apply configuration settings
    4. Initialize services
    5. Record deployment outcomes
    
    Output a structured report of the deployment execution, including the deployment URL.
    """,
    input_schema=None,
    output_key="deployment_result",
)

verifier = LlmAgent(
    name="Verifier",
    model=LiteLlm(model=MODEL_OLLAMA, model_type="openai", api_key="ollama-secret-key", api_base='http://localhost:11434/v1'),
    instruction="""You are a deployment verification specialist. Verify the deployment described in session state
    with key 'deployment_result'.
    
    Your verification should check:
    1. Accessibility of the deployed website
    2. Functionality of key features
    3. Performance metrics
    4. Security considerations
    5. Mobile responsiveness
    
    Output a verification report with a 'success' boolean, and if not successful, a list of 'issues'
    that need to be addressed. If fixes are needed, set 'needs_fixes' to true.
    """,
    input_schema=None,
    output_key="verification_result",
)

# --- Create the custom agent instance ---
startup_deployment_agent = StartupDeploymentAgent(
    name="StartupDeploymentAgent",
    package_analyzer=package_analyzer,
    deployment_preparer=deployment_preparer,
    infrastructure_configurator=infrastructure_configurator,
    deployer=deployer,
    verifier=verifier,
)

# --- Setup Runner and Session ---
session_service = InMemorySessionService()
initial_state = {}  # Empty initial state, will be populated with website package
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initial_state
)

runner = Runner(
    agent=startup_deployment_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# --- Function to Interact with the Agent ---
def call_deployment_agent(website_package: str, hosting_preference: str = "netlify"):
    """
    Sends a website package to the deployment agent and runs the deployment workflow.
    
    Args:
        website_package: The website package from the coding agent
        hosting_preference: Preferred hosting platform (netlify, vercel, github-pages, etc.)
    """
    current_session = session_service.get_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )
    
    if not current_session:
        logger.error("Session not found!")
        return

    # Update the session with the website package and hosting preference
    current_session.state["website_package"] = website_package
    current_session.state["hosting_preference"] = hosting_preference
    logger.info(f"Updated session with website package and hosting preference: {hosting_preference}")

    # Create message content
    content = types.Content(
        role='user', 
        parts=[types.Part(text=f"Deploy this website to {hosting_preference}.")]
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
    print("\n--- Deployment Result ---")
    print("Final Response: ", final_response)

    # Get final session state
    final_session = session_service.get_session(
        app_name=APP_NAME, 
        user_id=USER_ID, 
        session_id=SESSION_ID
    )
    
    # Return the deployment URL or result
    deployment_result = final_session.state.get("deployment_result", {})
    deployment_url = deployment_result.get("url", "No URL available")
    
    print(f"Deployment URL: {deployment_url}")
    return deployment_url

# --- Example of the complete end-to-end workflow ---
def complete_workflow(domain: str, hosting_platform: str = "netlify"):
    """
    Demonstrates the complete workflow from ideation to coding to deployment.
    """
    # Import the other agents
    from startup_ideation_agent import call_ideation_agent
    from startup_coding_agent import call_coding_agent
    
    # Step 1: Run the ideation agent
    print(f"Starting ideation process for domain: {domain}")
    startup_concept = call_ideation_agent(domain)
    
    # Step 2: Pass the result to the coding agent
    print("Generating website based on the startup concept...")
    website_package = call_coding_agent(startup_concept)
    
    # Step 3: Pass the website to the deployment agent
    print(f"Deploying website to {hosting_platform}...")
    deployment_url = call_deployment_agent(website_package, hosting_platform)
    
    print(f"\nComplete workflow finished. Your startup website is live at: {deployment_url}")
    return {
        "concept": startup_concept,
        "website_package": website_package,
        "deployment_url": deployment_url
    }

# --- Example usage ---
if __name__ == "__main__":
    # This would normally get the package from the coding agent
    example_package = """
    {
      "html": "<!DOCTYPE html><html>...</html>",
      "css": "body { ... }",
      "js": "document.addEventListener('DOMContentLoaded', function() { ... });",
      "assets": ["logo.png", "hero-bg.jpg"],
      "file_structure": {
        "index.html": "Main landing page",
        "css/style.css": "Main stylesheet",
        "js/main.js": "Main JavaScript file",
        "assets/": "Directory for images and other assets"
      }
    }
    """
    
    # Run the deployment agent with the example package
    deployment_url = call_deployment_agent(example_package, "netlify")