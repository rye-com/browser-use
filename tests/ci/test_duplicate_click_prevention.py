"""
Test file for verifying duplicate click prevention in agent's multi_act.

This test creates a simple HTML page with buttons, uses a mock LLM that tries to
click the same button twice, and verifies that the second click is blocked.

Usage:
    uv run pytest tests/ci/test_duplicate_click_prevention.py -v -s
"""

import asyncio

import pytest
from pytest_httpserver import HTTPServer

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistory, AgentHistoryList, BrowserStateHistory
from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.views import TabInfo
from browser_use.dom.views import DOMInteractedElement, DOMRect, NodeType
from tests.ci.conftest import create_mock_llm


@pytest.fixture(scope='session')
def http_server():
	"""Create and provide a test HTTP server that serves static content."""
	server = HTTPServer()
	server.start()

	# Add route for duplicate click test page
	server.expect_request('/duplicate-click-test').respond_with_data(
		"""
		<!DOCTYPE html>
		<html>
		<head>
			<title>Duplicate Click Test</title>
			<style>
				button {
					padding: 10px 20px;
					margin: 10px;
					cursor: pointer;
				}
				#result {
					margin-top: 20px;
					padding: 10px;
					border: 1px solid #ccc;
					min-height: 50px;
				}
			</style>
		</head>
		<body>
			<h1>Duplicate Click Test</h1>
			<button id="addToCartBtn" onclick="addToCart()">Add to Cart</button>
			<button id="otherBtn" onclick="doOther()">Other Action</button>
			<div id="result">Click count: 0</div>
			<div id="cartCount">Cart: 0 items</div>

			<script>
				let clickCount = 0;
				let cartItems = 0;

				function addToCart() {
					clickCount++;
					cartItems++;
					document.getElementById('result').textContent = 'Click count: ' + clickCount;
					document.getElementById('cartCount').textContent = 'Cart: ' + cartItems + ' items';
				}

				function doOther() {
					clickCount++;
					document.getElementById('result').textContent = 'Other clicked, count: ' + clickCount;
				}
			</script>
		</body>
		</html>
		""",
		content_type='text/html',
	)

	yield server
	server.stop()


@pytest.fixture(scope='session')
def base_url(http_server):
	"""Return the base URL for the test HTTP server."""
	return f'http://{http_server.host}:{http_server.port}'


@pytest.fixture(scope='module')
async def browser_session():
	"""Create and provide a Browser instance."""
	browser_session = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=True,
		)
	)
	await browser_session.start()
	yield browser_session
	await browser_session.kill()


class TestAgentHistoryGetClickedElements:
	"""Test cases for AgentHistoryList.get_clicked_element_hashes method."""

	def test_get_clicked_element_hashes_empty_history(self):
		"""Test that empty history returns empty set."""
		history = AgentHistoryList(history=[])
		clicked = history.get_clicked_element_hashes()
		assert clicked == set()

	def test_get_clicked_element_hashes_with_click_action(self):
		"""Test that click actions are tracked correctly."""
		from browser_use.agent.views import AgentOutput
		from browser_use.tools.registry.views import ActionModel

		# Create a simple action model for testing
		class TestActionModel(ActionModel):
			click: dict | None = None
			navigate: dict | None = None

		# Create a mock interacted element
		interacted_element = DOMInteractedElement(
			node_id=1,
			backend_node_id=100,
			frame_id='frame1',
			node_type=NodeType.ELEMENT_NODE,
			node_value='',
			node_name='BUTTON',
			attributes={'id': 'testBtn'},
			bounds=DOMRect(x=10, y=10, width=100, height=50),
			x_path='/html/body/button',
			element_hash=12345678,
		)

		# Create a mock history entry with a click action
		history_entry = AgentHistory(
			model_output=AgentOutput(
				thinking='Clicking button',
				evaluation_previous_goal='Success',
				memory='Clicked button',
				next_goal='Continue',
				action=[TestActionModel(click={'index': 1})],
			),
			result=[],
			state=BrowserStateHistory(
				url='http://test.com',
				title='Test',
				tabs=[TabInfo(tab_id='1234', url='http://test.com', title='Test')],
				interacted_element=[interacted_element],
			),
		)

		history = AgentHistoryList(history=[history_entry])
		clicked = history.get_clicked_element_hashes()

		assert len(clicked) == 1
		assert 12345678 in clicked

	def test_get_clicked_element_hashes_ignores_non_click_actions(self):
		"""Test that non-click actions are not tracked."""
		from browser_use.agent.views import AgentOutput
		from browser_use.tools.registry.views import ActionModel

		class TestActionModel(ActionModel):
			click: dict | None = None
			navigate: dict | None = None

		interacted_element = DOMInteractedElement(
			node_id=1,
			backend_node_id=100,
			frame_id='frame1',
			node_type=NodeType.ELEMENT_NODE,
			node_value='',
			node_name='A',
			attributes={'href': '/page'},
			bounds=DOMRect(x=10, y=10, width=100, height=50),
			x_path='/html/body/a',
			element_hash=87654321,
		)

		# Create a history entry with a navigate action (not click)
		history_entry = AgentHistory(
			model_output=AgentOutput(
				thinking='Navigating',
				evaluation_previous_goal='Success',
				memory='Navigated',
				next_goal='Continue',
				action=[TestActionModel(navigate={'url': 'http://test.com'})],
			),
			result=[],
			state=BrowserStateHistory(
				url='http://test.com',
				title='Test',
				tabs=[TabInfo(tab_id='1234', url='http://test.com', title='Test')],
				interacted_element=[interacted_element],
			),
		)

		history = AgentHistoryList(history=[history_entry])
		clicked = history.get_clicked_element_hashes()

		# Navigate action should not be tracked
		assert len(clicked) == 0

	def test_get_clicked_element_hashes_with_dict_interacted_element(self):
		"""Test that dict-format interacted elements are handled correctly."""
		from browser_use.agent.views import AgentOutput
		from browser_use.tools.registry.views import ActionModel

		class TestActionModel(ActionModel):
			click: dict | None = None

		# Create a dict representation of interacted element (as it might appear from serialization)
		interacted_element_dict = {
			'node_id': 1,
			'backend_node_id': 100,
			'frame_id': 'frame1',
			'node_type': 1,
			'node_value': '',
			'node_name': 'BUTTON',
			'attributes': {'id': 'testBtn'},
			'x_path': '/html/body/button',
			'element_hash': 99999999,
		}

		history_entry = AgentHistory(
			model_output=AgentOutput(
				thinking='Clicking',
				evaluation_previous_goal='Success',
				memory='Clicked',
				next_goal='Continue',
				action=[TestActionModel(click={'index': 1})],
			),
			result=[],
			state=BrowserStateHistory(
				url='http://test.com',
				title='Test',
				tabs=[TabInfo(tab_id='1234', url='http://test.com', title='Test')],
				interacted_element=[interacted_element_dict],  # type: ignore
			),
		)

		history = AgentHistoryList(history=[history_entry])
		clicked = history.get_clicked_element_hashes()

		assert len(clicked) == 1
		assert 99999999 in clicked


class TestDuplicateClickPrevention:
	"""Test cases for duplicate click prevention in agent's multi_act."""

	async def test_duplicate_click_returns_error(self, browser_session, base_url):
		"""Test that clicking the same element twice returns an error on the second click."""
		# Navigate to the test page first
		from browser_use.browser.events import NavigateToUrlEvent

		nav_event = browser_session.event_bus.dispatch(
			NavigateToUrlEvent(url=f'{base_url}/duplicate-click-test')
		)
		await nav_event
		await asyncio.sleep(0.5)

		# Get browser state to find the button
		state = await browser_session.get_browser_state_summary()
		selector_map = state.dom_state.selector_map

		# Find the "Add to Cart" button
		cart_btn_index = None
		for idx, element in selector_map.items():
			if element.attributes and element.attributes.get('id') == 'addToCartBtn':
				cart_btn_index = idx
				break

		assert cart_btn_index is not None, 'Could not find Add to Cart button'

		# Create mock LLM that clicks the same button twice
		mock_actions = [
			f"""
			{{
				"thinking": "Clicking add to cart button",
				"evaluation_previous_goal": "Starting task",
				"memory": "Found add to cart button",
				"next_goal": "Click add to cart",
				"action": [
					{{
						"click": {{
							"index": {cart_btn_index}
						}}
					}}
				]
			}}
			""",
			f"""
			{{
				"thinking": "Clicking add to cart button again",
				"evaluation_previous_goal": "Clicked once",
				"memory": "Need to click again",
				"next_goal": "Click add to cart again",
				"action": [
					{{
						"click": {{
							"index": {cart_btn_index}
						}}
					}}
				]
			}}
			""",
		]

		mock_llm = create_mock_llm(actions=mock_actions)

		# Create agent
		agent = Agent(
			task=f'Go to {base_url}/duplicate-click-test and click the Add to Cart button twice',
			llm=mock_llm,
			browser_session=browser_session,
		)

		# Run agent for enough steps (navigate + 2 click attempts + done)
		history = await agent.run(max_steps=5)

		# Check that the second step resulted in an error about duplicate click
		# Note: history[0] is the initial navigate action (auto-added by agent)
		# history[1] is the first click, history[2] is the duplicate click attempt
		assert len(history.history) >= 3, 'Expected at least 3 history entries (navigate + 2 clicks)'

		# First click should succeed (history[1] after the navigate)
		first_click_result = history.history[1].result[0] if history.history[1].result else None
		assert first_click_result is not None, 'First click should have a result'
		assert first_click_result.error is None, f'First click should succeed, got error: {first_click_result.error}'

		# Second click should have an error about duplicate (history[2])
		second_click_result = history.history[2].result[0] if history.history[2].result else None
		assert second_click_result is not None, 'Second click should have a result'
		assert second_click_result.error is not None, 'Second click should have an error'
		assert 'already clicked' in second_click_result.error.lower(), (
			f'Error should mention element was already clicked, got: {second_click_result.error}'
		)

	async def test_different_elements_can_be_clicked(self, browser_session, base_url):
		"""Test that clicking different elements is allowed."""
		# Navigate to the test page
		from browser_use.browser.events import NavigateToUrlEvent

		nav_event = browser_session.event_bus.dispatch(
			NavigateToUrlEvent(url=f'{base_url}/duplicate-click-test')
		)
		await nav_event
		await asyncio.sleep(0.5)

		# Get browser state
		state = await browser_session.get_browser_state_summary()
		selector_map = state.dom_state.selector_map

		# Find both buttons
		cart_btn_index = None
		other_btn_index = None
		for idx, element in selector_map.items():
			if element.attributes:
				if element.attributes.get('id') == 'addToCartBtn':
					cart_btn_index = idx
				elif element.attributes.get('id') == 'otherBtn':
					other_btn_index = idx

		assert cart_btn_index is not None, 'Could not find Add to Cart button'
		assert other_btn_index is not None, 'Could not find Other button'

		# Create mock LLM that clicks two different buttons
		mock_actions = [
			f"""
			{{
				"thinking": "Clicking add to cart button",
				"evaluation_previous_goal": "Starting task",
				"memory": "Found add to cart button",
				"next_goal": "Click add to cart",
				"action": [
					{{
						"click": {{
							"index": {cart_btn_index}
						}}
					}}
				]
			}}
			""",
			f"""
			{{
				"thinking": "Clicking other button",
				"evaluation_previous_goal": "Clicked cart button",
				"memory": "Now clicking other button",
				"next_goal": "Click other button",
				"action": [
					{{
						"click": {{
							"index": {other_btn_index}
						}}
					}}
				]
			}}
			""",
		]

		mock_llm = create_mock_llm(actions=mock_actions)

		# Create agent
		agent = Agent(
			task=f'Go to {base_url}/duplicate-click-test and click both buttons',
			llm=mock_llm,
			browser_session=browser_session,
		)

		# Run agent for enough steps
		history = await agent.run(max_steps=5)

		# Check that both clicks succeeded
		# Note: history[0] is the initial navigate action (auto-added by agent)
		# history[1] is the first click, history[2] is the second click
		assert len(history.history) >= 3, 'Expected at least 3 history entries (navigate + 2 clicks)'

		# Both clicks should succeed since they're different elements
		first_click_result = history.history[1].result[0] if history.history[1].result else None
		assert first_click_result is not None, 'First click should have a result'
		assert first_click_result.error is None, f'First click should succeed, got error: {first_click_result.error}'

		second_click_result = history.history[2].result[0] if history.history[2].result else None
		assert second_click_result is not None, 'Second click should have a result'
		assert second_click_result.error is None, f'Second click should succeed (different element), got error: {second_click_result.error}'
