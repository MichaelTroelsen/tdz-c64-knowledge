"""Phase 3: Temporal Analysis Tools. Split out of handlers.py (R12 follow-up).

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""


from mcp.types import TextContent


def handle_extract_document_events(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    min_confidence = arguments.get("min_confidence", 0.5)

    try:
        result = kb.extract_document_events(doc_id, min_confidence=min_confidence)

        output = f"# Event Extraction Complete\n\n"
        output += f"**Document:** {result['title']}\n"
        output += f"**Statistics:**\n"
        output += f"- Total events detected: {result['event_count']}\n"
        output += f"- Events with confidence >= {min_confidence}: {result['filtered_count']}\n"
        output += f"- Events stored to database: {result['stored_count']}\n\n"

        if result['events']:
            output += f"**Extracted Events:**\n\n"
            for i, event in enumerate(result['events'][:10], 1):
                date_str = event['date_info']['text'] if event['date_info'] else 'No date'
                output += f"{i}. **[{event['type']}]** {event['title'][:80]}...\n"
                output += f"   - Date: {date_str}\n"
                output += f"   - Confidence: {event['confidence']:.2f}\n"
                if event['entities']:
                    output += f"   - Entities: {', '.join(event['entities'][:3])}\n"
                output += "\n"

            if len(result['events']) > 10:
                output += f"... and {len(result['events']) - 10} more events\n\n"

        output += f"Events stored to database. Use `get_timeline` or `search_events_by_date` to query.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error extracting events: {str(e)}")]


def handle_get_timeline(kb, name: str, arguments: dict) -> list[TextContent]:
    start_year = arguments.get("start_year")
    end_year = arguments.get("end_year")
    category = arguments.get("category")
    min_importance = arguments.get("min_importance", 1)
    limit = arguments.get("limit")

    try:
        timeline = kb.get_timeline(
            start_year=start_year,
            end_year=end_year,
            category=category,
            min_importance=min_importance,
            limit=limit
        )

        if not timeline:
            # Try to build timeline if empty
            kb.build_timeline()
            timeline = kb.get_timeline(
                start_year=start_year,
                end_year=end_year,
                category=category,
                min_importance=min_importance,
                limit=limit
            )

        if not timeline:
            return [TextContent(type="text", text="No timeline entries found. Extract events from documents first using `extract_document_events`.")]

        output = f"# Timeline\n\n"

        # Add filter info
        filters = []
        if start_year:
            filters.append(f"from {start_year}")
        if end_year:
            filters.append(f"to {end_year}")
        if category:
            filters.append(f"category: {category}")
        if min_importance > 1:
            filters.append(f"importance >= {min_importance}")

        if filters:
            output += f"**Filters:** {', '.join(filters)}\n"

        output += f"**Total entries:** {len(timeline)}\n\n"
        output += f"**Timeline Entries:**\n\n"

        for entry in timeline[:50]:  # Limit to 50 entries
            importance_stars = "⭐" * entry['importance']
            output += f"**[{entry['display_date']}]** {entry['title'][:70]}...\n"
            output += f"  - Type: {entry['event_type']} | Importance: {importance_stars} ({entry['importance']}/5)\n"
            output += f"  - Confidence: {entry['confidence']:.2f}\n\n"

        if len(timeline) > 50:
            output += f"... and {len(timeline) - 50} more entries (use `limit` parameter to see more)\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting timeline: {str(e)}")]


def handle_search_events_by_date(kb, name: str, arguments: dict) -> list[TextContent]:
    start_year = arguments.get("start_year")
    end_year = arguments.get("end_year")
    event_type = arguments.get("event_type")
    min_confidence = arguments.get("min_confidence", 0.5)

    try:
        events = kb.search_events_by_date(
            start_year=start_year,
            end_year=end_year,
            event_type=event_type,
            min_confidence=min_confidence
        )

        if not events:
            return [TextContent(type="text", text="No events found matching the criteria.")]

        output = f"# Event Search Results\n\n"

        # Add search criteria
        criteria = []
        if start_year and end_year:
            criteria.append(f"{start_year}-{end_year}")
        elif start_year:
            criteria.append(f"from {start_year}")
        elif end_year:
            criteria.append(f"to {end_year}")
        if event_type:
            criteria.append(f"type: {event_type}")
        criteria.append(f"confidence >= {min_confidence}")

        output += f"**Search criteria:** {', '.join(criteria)}\n"
        output += f"**Results:** {len(events)} events\n\n"

        # Group events by year
        events_by_year = {}
        for event in events:
            year = event['year']
            if year not in events_by_year:
                events_by_year[year] = []
            events_by_year[year].append(event)

        for year in sorted(events_by_year.keys()):
            year_events = events_by_year[year]
            output += f"## {year} ({len(year_events)} events)\n\n"

            for event in year_events[:20]:  # Limit per year
                output += f"- **[{event['date_normalized']}]** {event['title'][:70]}...\n"
                output += f"  Type: {event['event_type']}, Confidence: {event['confidence']:.2f}\n"

            if len(year_events) > 20:
                output += f"  ... and {len(year_events) - 20} more events from {year}\n"

            output += "\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error searching events: {str(e)}")]


def handle_get_historical_context(kb, name: str, arguments: dict) -> list[TextContent]:
    year = arguments.get("year")
    context_years = arguments.get("context_years", 2)

    try:
        context = kb.get_historical_context(year, context_years=context_years)

        if context['total_events'] == 0:
            return [TextContent(type="text", text=f"No events found in the period {context['year_range'][0]}-{context['year_range'][1]}.")]

        output = f"# Historical Context for {year}\n\n"
        output += f"**Context range:** {context['year_range'][0]} - {context['year_range'][1]}\n"
        output += f"**Total events:** {context['total_events']}\n\n"

        # Show events by year
        for event_year in sorted(context['events_by_year'].keys()):
            year_events = context['events_by_year'][event_year]
            marker = "📍 **TARGET YEAR**" if event_year == year else ""

            output += f"## {event_year} ({len(year_events)} events) {marker}\n\n"

            for event in year_events[:10]:
                output += f"- **[{event['event_type']}]** {event['title'][:70]}...\n"
                output += f"  Date: {event['date_normalized']}, Confidence: {event['confidence']:.2f}\n"

            if len(year_events) > 10:
                output += f"  ... and {len(year_events) - 10} more events\n"

            output += "\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting historical context: {str(e)}")]


HANDLERS_TEMPORAL = {
    "extract_document_events": handle_extract_document_events,
    "get_timeline": handle_get_timeline,
    "search_events_by_date": handle_search_events_by_date,
    "get_historical_context": handle_get_historical_context,
}
