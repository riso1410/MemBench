def total_pages(item_count: int, page_size: int) -> int:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return item_count // page_size
