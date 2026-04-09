import json
import re

def sentence_match(text1, text2):
    # 1. Clean and split: removes punctuation and converts to lowercase
    def clean(text):
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    words1 = clean(text1)
    words2 = clean(text2)
    print(words1 , '\n')
    print(words2 , '\n')
    max_match_length = 0
    
    # 2. Traverse both word lists
    for i in range(len(words1)):
        for j in range(len(words2)):
            # If a word matches, start the counter
            if words1[i] == words2[j]:
                current_length = 0
                # Third loop: Check how long the sequence continues
                while (i + current_length < len(words1) and 
                       j + current_length < len(words2) and 
                       words1[i + current_length] == words2[j + current_length]):
                    current_length += 1
                
                # Update the best match found so far
                if current_length > max_match_length:
                    max_match_length = current_length
                
                # Early Exit
                if max_match_length > 3:
                    return True
                    
    return max_match_length > 3


def search_for_text(book_Jason , text , page_number):
    with open(book_Jason , mode="r") as read_file:
        book_data = json.load(read_file)

    desired_page = None
    page_index = None

    # 1️⃣ Find the page + BREAK
    for page_i, page in enumerate(book_data['pages']):
        if page['page'] == page_number: 
            desired_page = page
            page_index = page_i
            break   # ✅ FIX 1

    # 2️⃣ Handle page not found
    if desired_page is None:
        return None   # ✅ FIX 2

    # 3️⃣ Normalize text for exact match
    def normalize(text):
        return re.sub(r'[^\w\s]', '', text.lower())

    normalized_text = normalize(text)

    para_index_list = []

    for i, para in enumerate(desired_page["paragraphs"]):
        normalized_para = normalize(para['paragraph'])

        # exact match (normalized)
        if normalized_text in normalized_para:
            print(f"Exact match found at {i}")  
            para_index_list = [i]
            break

        # partial match
        if sentence_match(para['paragraph'], text):
            para_index_list.append(i)

    if len(para_index_list) == 1:
        return {
            "page_index": page_index,
            "para_indexs": para_index_list,
            "match_type": "exact_match"
        }
    elif len(para_index_list) > 1:
        return {
            "page_index": page_index,
            "para_indexs": para_index_list,
            "match_type": "partial_match"
        }

    return None