import wikipedia

def diseaseDetail(term):
    """
    Fetches a brief summary of the disease from Wikipedia and adds a URL.
    Returns an object with name, summary, url.
    """
    try:
        page = wikipedia.page(term)
        return {
            "name": term,
            "summary": page.summary,
            "url": page.url
        }
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0])
            return {
                "name": term,
                "summary": page.summary,
                "url": page.url
            }
        except:
            return {
                "name": term,
                "summary": "No details found.",
                "url": None
            }
    except:
        return {
            "name": term,
            "summary": "No details found.",
            "url": None
        }
