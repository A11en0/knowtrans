
# 企业匹配数据集的指令
ENTERPRICE_OLD_INSTRUCTION = "Are Enterprise A and Enterprise B the same Enterprise?\nChoose your answer from: [Yes, No]"
ENTERPRICE_NEW_INSTRUCTION = "Are Enterprise A and Enterprise B the same Enterprise?\nProvide a detailed reasoning that explains how to arrive at the answer.\nAfter your reasoning, provide your final answer in a separate line in the format of \"Final answer: Yes / No\"."
ENTERPRICE_OLD_KNOWLEDGE = "Note that missing values (N/A or \"nan\") should not be used as a basis for your decision."
ENTERPRICE_NEW_KNOWLEDGE = '''Guidelines:
1. **Name Similarity**: Consider enterprises as the same if one is a branch or subsidiary of the other. Names should indicate a clear relationship, such as inclusion of the parent company's name with additional identifiers for branches or subsidiaries.
2. **Registration Location**: Exact match in registration locations is not necessary. Different addresses can be present for branches or subsidiaries.
3. **Legal Person Name**: Differences in legal person names do not automatically imply the enterprises are different, as branches or subsidiaries can have different legal representatives.
4. **Business Scope**: Differences in business scope are acceptable as branches or subsidiaries might operate in different domains.
5. **Final Decision**: If the enterprise names suggest a parent-branch or parent-subsidiary relationship and the legal representative or registration location varies, consider them the same enterprise. If names are different without a clear relational indicator, consider them different enterprises.
'''

# abt-buy 
ABT_BUY_OLD_INSTRUCTION = "Are Product A and Product B the same Product?\nChoose your answer from: [Yes, No]"
ABT_BUY_NEW_INSTRUCTION = "Are Product A and Product B the same Product?\nProvide a detailed reasoning that explains how to arrive at the answer.\nAfter your reasoning, provide your final answer in a separate line in the format of \"Final answer: Yes / No\"."
ABT_BUY_OLD_KNOWLEDGE = "Note that missing values (N/A or \"nan\") should not be used as a basis for your decision."
# ABT_BUY_NEW_KNOWLEDGE = '''When comparing the two products, focus on matching key attributes such as product names, model numbers, and brands. Look for variations in spelling, abbreviations, and formatting that could still indicate the same product. For missing values like 'nan' or 'N/A', rely more heavily on the available information to make the best comparison. Ensure that both products belong to the same category and consider the context of the prices, but do not rely solely on price differences to determine whether they are the same product. Always provide a clear and logical reasoning to support your final decision.'''
# ABT_BUY_NEW_KNOWLEDGE = '''When determining if two products are the same, prioritize matching key elements such as the brand, product name, and model number, as these are strong indicators of product identity. Be aware that missing values like 'nan' or 'N/A' indicate incomplete data, so rely more heavily on the available information. Look for common patterns in abbreviations, synonyms, and alternate spellings. Small differences in product names or descriptions may still refer to the same item. Price differences may offer clues but should not solely determine the final decision if other attributes align.'''
# ABT_BUY_NEW_KNOWLEDGE = '''When comparing Product A and Product B, focus on the keywords and specific details provided in the product names. Consider aspects such as model numbers, series, compatibility, and distinguishing features mentioned in the descriptions. Even if a description is missing (N/A or "nan"), cross-reference the available data points, especially unique identifiers and key characteristics, to determine if the products are indeed the same. Ignore generic terms and concentrate on exact matches or closely related specifics like model numbers or unique attributes that might indicate the products are identical.'''
ABT_BUY_NEW_KNOWLEDGE = '''When comparing products, focus on the specific details within both the product names and descriptions. Look for key characteristics such as model numbers, color, capacity, and compatibility. Pay attention to unique identifiers and technical specifications that can help distinguish one product from another. Remember to disregard missing values (N/A or "nan") in the descriptions as these should not influence your decision. Instead, rely on the available information to determine if there are significant similarities or differences between the products. Comparing details like brand, product type, and specific features can help you accurately decide whether the products are the same or different.'''

# walmart-amazon
WALMART_AMAZON_OLD_INSTRUCTION = "Are Product A and Product B the same Product?\nChoose your answer from: [Yes, No]"
WALMART_AMAZON_NEW_INSTRUCTION = "Are Product A and Product B the same Product?\nProvide a detailed reasoning that explains how to arrive at the answer.\nAfter your reasoning, provide your final answer in a separate line in the format of \"Final answer: Yes / No\"."
WALMART_AMAZON_OLD_KNOWLEDGE = "Note that missing values (N/A or \"nan\") should not be used as a basis for your decision."
WALMART_AMAZON_NEW_KNOWLEDGE = '''When comparing two products, prioritize matching key elements such as the brand, product name, and model number, as these are strong indicators of product identity. Be aware that missing values like 'nan' or 'N/A' indicate incomplete data, so rely more heavily on the available information. Look for common patterns in abbreviations, synonyms, and alternate spellings. Small differences in product names or descriptions may still refer to the same item. Price differences may offer clues but should not solely determine the final decision if other attributes align.'''
        



