# test_main.py
import unittest
from unittest.mock import patch, MagicMock
from main import product_identifier_func, find_budget_friendly_option

# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

class TestBusinessLogic(unittest.TestCase):

    @patch('main.OpenAIGenerator')
    def test_product_identifier_func(self, mock_openai_generator):
        # Mock the OpenAIGenerator to return a response
        mock_response = {'llm': {'replies': ['["product1", "product2"]']}}
        mock_openai_generator.return_value.run.return_value = mock_response

        # Test the function
        result = product_identifier_func('What are the products?')
        self.assertEqual(result, {'product1': {}, 'product2': {}})

    @patch('main.OpenAIGenerator')
    def test_product_identifier_func_with_exception(self, mock_openai_generator):
        # Mock the OpenAIGenerator to raise an exception
        mock_openai_generator.return_value.run.side_effect = Exception('Mocked exception')

        # Test the function
        result = product_identifier_func('What are the products?')
        self.assertEqual(result, 'Got an exception finding product list. No product found.')

    def test_find_budget_friendly_option(self):
        # Test the function with a single product
        products = {'product1': {'price': 10.0, 'name': 'Product 1', 'url': 'https://example.com/product1'}}
        result = find_budget_friendly_option(products)
        self.assertEqual(result, {'product1': {'price': 10.0, 'name': 'Product 1', 'url': 'https://example.com/product1'}})

        # Test the function with multiple products
        products = {'product1': [{'price': 10.0, 'name': 'Product 1', 'url': 'https://example.com/product1'},
                                  {'price': 20.0, 'name': 'Product 2', 'url': 'https://example.com/product2'}]}
        result = find_budget_friendly_option(products)
        self.assertEqual(result, {'product1': {'price': 10.0, 'name': 'Product 1', 'url': 'https://example.com/product1'}})

if __name__ == '__main__':
    unittest.main()