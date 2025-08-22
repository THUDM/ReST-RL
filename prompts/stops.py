# stop strings for different domains

def get_stop_strings(domain: str = None) -> list[str] | None:
    if domain == 'BigCodeBench':
        return ["```"]
    elif domain == 'DS1000':
        return ["</code>", "### SOLUTION END", "### END SOLUTION", "# SOLUTION END", "# END SOLUTION", "END SOLUTION",
                "SOLUTION END"]
    elif domain == 'APPS':
        return ["```"]
    else:
        raise ValueError("Invalid domain")
