def get_context(search_output, book_obj, target_text):
    """
    Returns context_paragraph and target_text based on search output.

    Args:
        search_output (dict): Output from search_for_text
        book_obj (dict): Full book object with pages and paragraphs
        target_text (str): User input text

    Returns:
        tuple: (context_paragraph, target_text)
    """
    page_index = search_output['page_index']
    para_indexs = search_output['para_indexs']
    match_type = search_output['match_type']

    page = book_obj['pages'][page_index]

    context_paragraph = ""

    if match_type == "exact_match":
        # Only one index for exact match
        i = para_indexs[0]

        # Previous paragraph if exists
        if i > 0:
            context_paragraph += page['paragraphs'][i - 1]['paragraph'] + " "

        # Current paragraph
        context_paragraph += page['paragraphs'][i]['paragraph'] + " "

        # Next paragraph if exists
        if i < len(page['paragraphs']) - 1:
            context_paragraph += page['paragraphs'][i + 1]['paragraph']

    elif match_type == "partial_match":
        # Concatenate all paragraphs in para_indexs
        for i in para_indexs:
            context_paragraph += page['paragraphs'][i]['paragraph'] + " "

    # Strip any trailing spaces
    context_paragraph = context_paragraph.strip()

    return context_paragraph, target_text