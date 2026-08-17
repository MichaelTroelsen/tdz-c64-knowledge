"""Date and event extraction, timelines and anomaly detection.

Extracted from server.py's KnowledgeBase by R12 step 2. These are mixins,
not standalone classes: every method still expects the attributes
KnowledgeBase.__init__ sets up, and nothing here imports server, so the
class can be assembled in server.py without an import cycle.
"""
from typing import Optional
import json


class TemporalMixin:

    def extract_dates_from_text(self, text: str) -> list[dict]:
        """
        Extract dates and temporal references from text using regex patterns.

        This method identifies various date formats commonly found in C64 documentation:
        - Years: "1982", "1983"
        - Month/Year: "January 1982", "Jan 1982"
        - Full dates: "January 15, 1982", "15-Jan-1982"
        - Decades: "1980s", "early 80s"
        - Date ranges: "1982-1985", "1982 to 1985"

        Args:
            text: Text to extract dates from

        Returns:
            List of dictionaries with:
            - 'text': Original matched text
            - 'type': Date type (year, month_year, full_date, decade, range)
            - 'year': Extracted year (int or None)
            - 'month': Extracted month (int or None)
            - 'day': Extracted day (int or None)
            - 'start_pos': Character position in text
            - 'end_pos': Character position in text

        Examples:
            >>> kb.extract_dates_from_text("The C64 was released in August 1982")
            [{'text': 'August 1982', 'type': 'month_year', 'year': 1982, 'month': 8, ...}]

            >>> kb.extract_dates_from_text("Popular throughout the 1980s")
            [{'text': '1980s', 'type': 'decade', 'year': 1980, ...}]
        """
        import re

        dates = []

        # Month names mapping
        months = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }

        # Pattern 1a: Full dates - Month Day, Year (January 15, 1982)
        pattern1a = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s]+(\d{1,2}),?[\s]+(\d{4})\b'
        for match in re.finditer(pattern1a, text, re.IGNORECASE):
            month_str = match.group(1).lower()
            day = int(match.group(2))
            year = int(match.group(3))
            month = months.get(month_str, None)

            dates.append({
                'text': match.group(0),
                'type': 'full_date',
                'year': year,
                'month': month,
                'day': day,
                'start_pos': match.start(),
                'end_pos': match.end()
            })

        # Pattern 1b: Full dates - Day Month Year (15 Jan 1982 | 15-Jan-1982)
        pattern1b = r'\b(\d{1,2})[\s\-](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,\-]+(\d{4})\b'
        for match in re.finditer(pattern1b, text, re.IGNORECASE):
            day = int(match.group(1))
            month_str = match.group(2).lower()
            year = int(match.group(3))
            month = months.get(month_str, None)

            # Skip if already matched
            if not any(d['start_pos'] <= match.start() < d['end_pos'] for d in dates):
                dates.append({
                    'text': match.group(0),
                    'type': 'full_date',
                    'year': year,
                    'month': month,
                    'day': day,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })

        # Pattern 2: Month Year (January 1982 | Jan 1982)
        pattern2 = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+(\d{4})\b'
        for match in re.finditer(pattern2, text, re.IGNORECASE):
            month_str = match.group(1).lower()
            year = int(match.group(2))
            month = months.get(month_str, None)

            # Skip if already matched as full date
            if not any(d['start_pos'] <= match.start() < d['end_pos'] for d in dates):
                dates.append({
                    'text': match.group(0),
                    'type': 'month_year',
                    'year': year,
                    'month': month,
                    'day': None,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })

        # Pattern 3: Year only (1982, 1983) - must be in C64 era (1975-2000)
        pattern3 = r'\b(19[7-9]\d|200\d)\b'
        for match in re.finditer(pattern3, text):
            year = int(match.group(0))

            # Skip if already matched in a more specific date
            if not any(d['start_pos'] <= match.start() < d['end_pos'] for d in dates):
                dates.append({
                    'text': match.group(0),
                    'type': 'year',
                    'year': year,
                    'month': None,
                    'day': None,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })

        # Pattern 4: Decades (1980s, early 80s, mid-80s)
        pattern4 = r'\b(early\s+|mid[\s\-]?|late\s+)?(19)?([7-9]0)s\b'
        for match in re.finditer(pattern4, text, re.IGNORECASE):
            decade = match.group(3)
            prefix = match.group(1) if match.group(1) else ''

            # Determine decade start year
            if match.group(2):  # Has "19" prefix
                year = int(f"19{decade}")
            else:
                year = int(f"19{decade}")  # Assume 1900s

            dates.append({
                'text': match.group(0),
                'type': 'decade',
                'year': year,
                'month': None,
                'day': None,
                'prefix': prefix.strip(),
                'start_pos': match.start(),
                'end_pos': match.end()
            })

        # Pattern 5: Date ranges (1982-1985, 1982 to 1985)
        pattern5 = r'\b(19[7-9]\d)[\s]*([\-to]+)[\s]*(19[7-9]\d)\b'
        for match in re.finditer(pattern5, text, re.IGNORECASE):
            start_year = int(match.group(1))
            end_year = int(match.group(3))

            dates.append({
                'text': match.group(0),
                'type': 'range',
                'year': start_year,
                'end_year': end_year,
                'month': None,
                'day': None,
                'start_pos': match.start(),
                'end_pos': match.end()
            })

        # Sort by position in text
        dates.sort(key=lambda x: x['start_pos'])

        return dates

    def normalize_date(self, date_dict: dict) -> tuple[str, int, int, int]:
        """
        Normalize a date dictionary to ISO format and component integers.

        Args:
            date_dict: Dictionary from extract_dates_from_text()

        Returns:
            Tuple of (iso_string, year, month, day)
            - iso_string: ISO 8601 date string (YYYY-MM-DD or YYYY-MM or YYYY)
            - year: Integer year
            - month: Integer month or 0 if not specified
            - day: Integer day or 0 if not specified

        Examples:
            >>> date_dict = {'type': 'month_year', 'year': 1982, 'month': 8, 'day': None}
            >>> kb.normalize_date(date_dict)
            ('1982-08', 1982, 8, 0)

            >>> date_dict = {'type': 'year', 'year': 1982, 'month': None, 'day': None}
            >>> kb.normalize_date(date_dict)
            ('1982', 1982, 0, 0)
        """
        year = date_dict.get('year', 0)
        month = date_dict.get('month', 0) if date_dict.get('month') else 0
        day = date_dict.get('day', 0) if date_dict.get('day') else 0

        # Build ISO string based on precision
        if day > 0 and month > 0:
            iso_string = f"{year:04d}-{month:02d}-{day:02d}"
        elif month > 0:
            iso_string = f"{year:04d}-{month:02d}"
        else:
            iso_string = f"{year:04d}"

        return iso_string, year, month, day

    def detect_events_in_text(self, text: str, doc_id: Optional[str] = None) -> list[dict]:
        """
        Detect significant events in text using pattern matching.

        This method identifies events commonly found in C64 documentation:
        - Product releases: "released", "launched", "introduced"
        - Company milestones: "founded", "acquired", "established"
        - Technical innovations: "first", "invented", "developed", "created"
        - Cultural events: "competition", "demo", "conference", "meeting"

        Args:
            text: Text to analyze for events
            doc_id: Optional document ID for context

        Returns:
            List of event dictionaries with:
            - 'type': Event type (release, milestone, innovation, cultural)
            - 'title': Brief event title
            - 'description': Event description with context
            - 'date_info': Associated date information
            - 'confidence': Confidence score (0.0-1.0)
            - 'entities': Related entities (if any)
            - 'position': Character position in text

        Examples:
            >>> kb.detect_events_in_text("The C64 was released in August 1982")
            [{'type': 'release', 'title': 'C64 released', 'date_info': {...}, ...}]
        """
        import re

        events = []

        # First, extract all dates from the text
        dates = self.extract_dates_from_text(text)

        # Event patterns with trigger words and types
        event_patterns = [
            # Product releases
            {
                'pattern': r'\b(released|launched|introduced|unveiled|announced|shipped|available)\b',
                'type': 'release',
                'confidence': 0.85
            },
            # Company milestones
            {
                'pattern': r'\b(founded|established|acquired|merged|created|formed|incorporated)\b',
                'type': 'milestone',
                'confidence': 0.80
            },
            # Technical innovations
            {
                'pattern': r'\b(first|invented|developed|created|designed|pioneered|innovated|breakthrough)\b',
                'type': 'innovation',
                'confidence': 0.75
            },
            # Cultural events
            {
                'pattern': r'\b(competition|contest|demo|demonstration|conference|convention|meeting|expo|exhibition|show)\b',
                'type': 'cultural',
                'confidence': 0.70
            },
            # Version/update events
            {
                'pattern': r'\b(updated|upgraded|revised|version|release|edition)\b',
                'type': 'update',
                'confidence': 0.65
            }
        ]

        # Find all event trigger words
        for event_pattern in event_patterns:
            for match in re.finditer(event_pattern['pattern'], text, re.IGNORECASE):
                trigger_word = match.group(0)
                trigger_pos = match.start()

                # Find the nearest date (within 200 characters)
                nearest_date = None
                min_distance = float('inf')

                for date in dates:
                    distance = abs(date['start_pos'] - trigger_pos)
                    if distance < min_distance and distance < 200:
                        min_distance = distance
                        nearest_date = date

                # Extract context around the trigger word (100 chars before and after)
                context_start = max(0, trigger_pos - 100)
                context_end = min(len(text), trigger_pos + 100)
                context = text[context_start:context_end].strip()

                # Try to extract a title from the surrounding sentence
                # Find sentence boundaries
                sentence_start = max(0, text.rfind('.', 0, trigger_pos) + 1)
                sentence_end = text.find('.', trigger_pos)
                if sentence_end == -1:
                    sentence_end = len(text)

                sentence = text[sentence_start:sentence_end].strip()

                # Create event title (first 100 chars of sentence or context)
                title = sentence[:100] if len(sentence) <= 100 else sentence[:97] + "..."

                # Adjust confidence based on date proximity
                confidence = event_pattern['confidence']
                if nearest_date:
                    # Higher confidence if date is very close
                    if min_distance < 50:
                        confidence = min(1.0, confidence + 0.1)
                else:
                    # Lower confidence if no date found
                    confidence = max(0.3, confidence - 0.2)

                # Try to extract entities from the context (basic noun extraction)
                entities = []
                # Look for capitalized words (potential proper nouns)
                entity_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', sentence)
                entities = list(set(entity_matches))[:5]  # Max 5 entities

                event = {
                    'type': event_pattern['type'],
                    'title': title,
                    'description': context,
                    'trigger_word': trigger_word,
                    'date_info': nearest_date,
                    'confidence': confidence,
                    'entities': entities,
                    'position': trigger_pos,
                    'sentence': sentence
                }

                events.append(event)

        # Sort by position in text
        events.sort(key=lambda x: x['position'])

        return events

    def detect_anomalies(self, min_severity: str = 'moderate', days: int = 7) -> dict:
        """
        Detect anomalies in document monitoring history.

        This method analyzes URL monitoring history to identify unusual patterns:
        - Unexpected update frequencies (sites changing much more/less than normal)
        - Performance degradation (significantly slower response times)
        - Change magnitude anomalies (unusually large or small content changes)

        The system learns baseline patterns for each document over a configurable
        learning period (default: 30 days) and scores deviations across multiple
        dimensions.

        Args:
            min_severity: Minimum severity level to include ('normal', 'minor', 'moderate', 'critical')
            days: Number of days of history to analyze (default: 7)

        Returns:
            Dictionary containing:
            - 'anomalies': List of anomaly records with scores and details
            - 'total_anomalies': Total number of anomalies found
            - 'by_severity': Count breakdown by severity level
            - 'avg_score': Average anomaly score
            - 'time_range_days': Days of history analyzed

        Severity Levels:
            - normal: 0-30 (no alert needed)
            - minor: 31-60 (include in digest)
            - moderate: 61-85 (immediate notification)
            - critical: 86-100 (urgent alert)

        Examples:
            >>> # Get critical and moderate anomalies from last 7 days
            >>> results = kb.detect_anomalies(min_severity='moderate', days=7)
            >>> print(f"Found {results['total_anomalies']} anomalies")
            >>> for anomaly in results['anomalies']:
            ...     print(f"  {anomaly['doc_title']}: {anomaly['severity']} ({anomaly['score']}/100)")

            >>> # Get all anomalies (including minor) from last 30 days
            >>> results = kb.detect_anomalies(min_severity='minor', days=30)
            >>> print(f"Breakdown: {results['by_severity']}")

        Note:
            Requires anomaly detection to be enabled (USE_ANOMALY_DETECTION=1).
            Will return error if anomaly_detector is not initialized.
        """
        if not self.anomaly_detector:
            return {
                'error': 'Anomaly detection not available',
                'anomalies': [],
                'total_anomalies': 0,
                'by_severity': {},
                'avg_score': 0.0,
                'time_range_days': days
            }

        try:
            # Get anomalies from detector
            anomalies = self.anomaly_detector.get_anomalies(
                min_severity=min_severity,
                days=days
            )

            # Calculate statistics
            total_anomalies = len(anomalies)
            by_severity = {}
            total_score = 0.0

            for anomaly in anomalies:
                severity = anomaly.get('severity', 'unknown')
                by_severity[severity] = by_severity.get(severity, 0) + 1
                total_score += anomaly.get('score', 0.0)

            avg_score = total_score / total_anomalies if total_anomalies > 0 else 0.0

            return {
                'anomalies': anomalies,
                'total_anomalies': total_anomalies,
                'by_severity': by_severity,
                'avg_score': round(avg_score, 2),
                'time_range_days': days
            }

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            return {
                'error': f'Anomaly detection failed: {str(e)}',
                'anomalies': [],
                'total_anomalies': 0,
                'by_severity': {},
                'avg_score': 0.0,
                'time_range_days': days
            }

    def extract_document_events(self, doc_id: str, min_confidence: float = 0.5) -> dict:
        """
        Extract events from a specific document and store them in the database.

        Args:
            doc_id: Document ID to process
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            Dictionary with:
            - 'doc_id': Document ID
            - 'title': Document title
            - 'event_count': Number of events extracted
            - 'events': List of extracted events
            - 'stored_count': Number of events stored to database

        Examples:
            >>> result = kb.extract_document_events('doc123', min_confidence=0.6)
            >>> print(f"Found {result['event_count']} events")
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document {doc_id} not found")

        doc = self.documents[doc_id]

        # Load document chunks
        chunks = self._get_chunks_db(doc_id)

        # Combine chunks into full text (with chunk boundaries marked)
        full_text = ""
        chunk_positions = []  # Track which chunk each character belongs to

        for chunk in chunks:
            start_pos = len(full_text)
            full_text += chunk.content + "\n\n"
            end_pos = len(full_text)
            chunk_positions.append({
                'chunk_id': chunk.chunk_id,
                'start': start_pos,
                'end': end_pos
            })

        # Detect events in the full text
        events = self.detect_events_in_text(full_text, doc_id)

        # Filter by confidence
        filtered_events = [e for e in events if e['confidence'] >= min_confidence]

        # Store events to database
        stored_count = 0
        cursor = self.db_conn.cursor()

        try:
            cursor.execute("BEGIN TRANSACTION")

            for event in filtered_events:
                # Generate event ID
                import uuid
                from datetime import datetime

                event_id = str(uuid.uuid4())

                # Normalize date if available
                date_normalized = None
                year = None
                month = None
                day = None

                if event['date_info']:
                    iso_string, year, month, day = self.normalize_date(event['date_info'])
                    date_normalized = iso_string

                # Store event
                cursor.execute("""
                    INSERT INTO events
                    (event_id, event_type, title, description, date_extracted,
                     date_normalized, year, month, day, confidence, entities,
                     metadata, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    event['type'],
                    event['title'],
                    event['description'],
                    event['date_info']['text'] if event['date_info'] else None,
                    date_normalized,
                    year if year else None,
                    month if month else None,
                    day if day else None,
                    event['confidence'],
                    json.dumps(event['entities']),
                    json.dumps({
                        'trigger_word': event['trigger_word'],
                        'sentence': event['sentence']
                    }),
                    datetime.utcnow().isoformat()
                ))

                # Find which chunk this event belongs to
                event_chunk_id = None
                for chunk_pos in chunk_positions:
                    if chunk_pos['start'] <= event['position'] < chunk_pos['end']:
                        event_chunk_id = chunk_pos['chunk_id']
                        break

                # Store document-event mapping
                mapping_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO document_events
                    (mapping_id, doc_id, event_id, context, position, created_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    mapping_id,
                    doc_id,
                    event_id,
                    event['description'],
                    event['position'],
                    datetime.utcnow().isoformat()
                ))

                stored_count += 1

            self.db_conn.commit()
            self.logger.info(f"Stored {stored_count} events for document {doc_id}")

        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Failed to store events: {e}")
            raise

        return {
            'doc_id': doc_id,
            'title': doc.title,
            'event_count': len(events),
            'filtered_count': len(filtered_events),
            'stored_count': stored_count,
            'events': filtered_events
        }

    def build_timeline(self, min_confidence: float = 0.5,
                       categorize: bool = True) -> dict:
        """
        Build a chronological timeline from events in the database.

        This method creates timeline_entries records from events, organizing them
        chronologically and optionally categorizing them by event type and time period.

        Args:
            min_confidence: Minimum confidence threshold for events to include
            categorize: If True, categorize events by decade and type

        Returns:
            Dictionary with:
            - 'total_events': Total events processed
            - 'timeline_entries': Number of timeline entries created
            - 'date_range': Earliest and latest years
            - 'categories': Event counts by category
            - 'by_year': Event counts by year

        Examples:
            >>> result = kb.build_timeline(min_confidence=0.6)
            >>> print(f"Created timeline with {result['timeline_entries']} entries")
        """
        import uuid
        from datetime import datetime

        cursor = self.db_conn.cursor()

        # Get all events with dates above confidence threshold
        events = cursor.execute("""
            SELECT event_id, event_type, title, description,
                   date_normalized, year, month, day, confidence
            FROM events
            WHERE confidence >= ? AND year IS NOT NULL
            ORDER BY year, month, day
        """, (min_confidence,)).fetchall()

        if not events:
            self.logger.warning("No events found with dates to build timeline")
            return {
                'total_events': 0,
                'timeline_entries': 0,
                'date_range': None,
                'categories': {},
                'by_year': {}
            }

        # Clear existing timeline entries (rebuild)
        cursor.execute("DELETE FROM timeline_entries")

        # Track statistics
        categories = {}
        by_year = {}
        created_count = 0

        try:
            for event in events:
                event_id, event_type, title, description, date_normalized, year, month, day, confidence = event

                # Create display date
                if day and month:
                    display_date = f"{year}-{month:02d}-{day:02d}"
                elif month:
                    display_date = f"{year}-{month:02d}"
                else:
                    display_date = str(year)

                # Create sort order (YYYYMMDD as integer)
                sort_order = year * 10000
                if month:
                    sort_order += month * 100
                if day:
                    sort_order += day

                # Determine category
                if categorize:
                    decade = (year // 10) * 10
                    category = f"{decade}s-{event_type}"
                else:
                    category = event_type

                # Determine importance (1-5 scale, 5 is highest)
                # Higher confidence = higher importance
                importance = min(5, max(1, int(confidence * 5)))

                # Create timeline entry
                entry_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO timeline_entries
                    (entry_id, event_id, display_date, sort_order,
                     category, importance, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id,
                    event_id,
                    display_date,
                    sort_order,
                    category,
                    importance,
                    datetime.utcnow().isoformat()
                ))

                created_count += 1

                # Update statistics
                categories[category] = categories.get(category, 0) + 1
                by_year[year] = by_year.get(year, 0) + 1

            self.db_conn.commit()
            self.logger.info(f"Built timeline with {created_count} entries from {len(events)} events")

        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Failed to build timeline: {e}")
            raise

        # Determine date range
        years = list(by_year.keys())
        date_range = (min(years), max(years)) if years else None

        return {
            'total_events': len(events),
            'timeline_entries': created_count,
            'date_range': date_range,
            'categories': categories,
            'by_year': by_year
        }

    def get_timeline(self, start_year: Optional[int] = None,
                     end_year: Optional[int] = None,
                     category: Optional[str] = None,
                     min_importance: int = 1,
                     limit: Optional[int] = None) -> list[dict]:
        """
        Query timeline entries with optional filtering.

        Args:
            start_year: Filter events from this year onwards (inclusive)
            end_year: Filter events up to this year (inclusive)
            category: Filter by category (e.g., "1980s-release")
            min_importance: Minimum importance level (1-5)
            limit: Maximum number of entries to return

        Returns:
            List of timeline entry dictionaries with:
            - 'entry_id': Timeline entry ID
            - 'event_id': Associated event ID
            - 'display_date': Formatted date string
            - 'sort_order': Integer for chronological sorting
            - 'category': Event category
            - 'importance': Importance level (1-5)
            - 'event_type': Event type
            - 'title': Event title
            - 'description': Event description
            - 'confidence': Event confidence

        Examples:
            >>> # Get all events from the 1980s
            >>> timeline = kb.get_timeline(start_year=1980, end_year=1989)

            >>> # Get important release events
            >>> releases = kb.get_timeline(category="1980s-release", min_importance=4)
        """
        cursor = self.db_conn.cursor()

        # Build query with filters
        query = """
            SELECT
                t.entry_id, t.event_id, t.display_date, t.sort_order,
                t.category, t.importance,
                e.event_type, e.title, e.description, e.confidence, e.year
            FROM timeline_entries t
            JOIN events e ON t.event_id = e.event_id
            WHERE t.importance >= ?
        """
        params = [min_importance]

        if start_year:
            query += " AND e.year >= ?"
            params.append(start_year)

        if end_year:
            query += " AND e.year <= ?"
            params.append(end_year)

        if category:
            query += " AND t.category = ?"
            params.append(category)

        query += " ORDER BY t.sort_order ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        results = cursor.execute(query, params).fetchall()

        timeline = []
        for row in results:
            timeline.append({
                'entry_id': row[0],
                'event_id': row[1],
                'display_date': row[2],
                'sort_order': row[3],
                'category': row[4],
                'importance': row[5],
                'event_type': row[6],
                'title': row[7],
                'description': row[8],
                'confidence': row[9],
                'year': row[10]
            })

        return timeline

    def search_events_by_date(self, start_year: Optional[int] = None,
                              end_year: Optional[int] = None,
                              event_type: Optional[str] = None,
                              min_confidence: float = 0.5) -> list[dict]:
        """
        Search for events within a date range.

        Args:
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            event_type: Filter by event type (release, milestone, innovation, etc.)
            min_confidence: Minimum confidence threshold

        Returns:
            List of event dictionaries ordered chronologically

        Examples:
            >>> # Find all releases in the 1980s
            >>> events = kb.search_events_by_date(
            ...     start_year=1980,
            ...     end_year=1989,
            ...     event_type='release'
            ... )
        """
        cursor = self.db_conn.cursor()

        query = """
            SELECT event_id, event_type, title, description,
                   date_extracted, date_normalized, year, month, day,
                   confidence, entities, metadata
            FROM events
            WHERE confidence >= ?
        """
        params = [min_confidence]

        if start_year:
            query += " AND year >= ?"
            params.append(start_year)

        if end_year:
            query += " AND year <= ?"
            params.append(end_year)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY year, month, day"

        results = cursor.execute(query, params).fetchall()

        events = []
        for row in results:
            events.append({
                'event_id': row[0],
                'event_type': row[1],
                'title': row[2],
                'description': row[3],
                'date_extracted': row[4],
                'date_normalized': row[5],
                'year': row[6],
                'month': row[7],
                'day': row[8],
                'confidence': row[9],
                'entities': json.loads(row[10]) if row[10] else [],
                'metadata': json.loads(row[11]) if row[11] else {}
            })

        return events

    def get_historical_context(self, year: int, context_years: int = 2) -> dict:
        """
        Get historical context for a specific year.

        Returns events from the target year plus surrounding years to provide context.

        Args:
            year: Target year
            context_years: Number of years before/after to include (default: 2)

        Returns:
            Dictionary with:
            - 'target_year': The requested year
            - 'year_range': (start_year, end_year)
            - 'events': List of events in the range
            - 'events_by_year': Events grouped by year
            - 'total_events': Total event count

        Examples:
            >>> # Get context for 1982 (the C64 release year)
            >>> context = kb.get_historical_context(1982, context_years=2)
            >>> # Returns events from 1980-1984
        """
        start_year = year - context_years
        end_year = year + context_years

        events = self.search_events_by_date(start_year=start_year, end_year=end_year)

        # Group events by year
        events_by_year = {}
        for event in events:
            event_year = event['year']
            if event_year not in events_by_year:
                events_by_year[event_year] = []
            events_by_year[event_year].append(event)

        return {
            'target_year': year,
            'year_range': (start_year, end_year),
            'events': events,
            'events_by_year': events_by_year,
            'total_events': len(events)
        }

    def visualize_timeline(self, start_year: Optional[int] = None,
                          end_year: Optional[int] = None,
                          output_path: str = "timeline.html") -> str:
        """
        Create interactive timeline visualization using Plotly.

        Generates a horizontal timeline showing events chronologically with:
        - Color-coded event types
        - Hover information with event details
        - Importance-based marker sizes
        - Zoomable and interactive

        Args:
            start_year: Filter events from this year (optional)
            end_year: Filter events to this year (optional)
            output_path: Path to save HTML file

        Returns:
            Path to saved HTML file

        Examples:
            >>> kb.visualize_timeline(start_year=1980, end_year=1990)
            'timeline.html'
        """
        import plotly.graph_objects as go
        from pathlib import Path

        # Get timeline entries
        timeline = self.get_timeline(start_year=start_year, end_year=end_year)

        if not timeline:
            self.logger.warning("No timeline entries to visualize")
            return ""

        # Color map for event types
        color_map = {
            'release': '#FF6B6B',      # Red
            'milestone': '#4ECDC4',    # Teal
            'innovation': '#95E1D3',   # Mint
            'cultural': '#FFA07A',     # Salmon
            'update': '#9B59B6'        # Purple
        }

        # Prepare data for plotting
        dates = []
        titles = []
        types = []
        importances = []
        confidences = []
        descriptions = []
        colors = []

        for entry in timeline:
            # Parse date for plotting
            date_str = entry['display_date']
            if len(date_str) == 4:  # Year only
                date_str += '-01-01'
            elif len(date_str) == 7:  # Year-month
                date_str += '-01'

            dates.append(date_str)
            titles.append(entry['title'][:100])
            types.append(entry['event_type'])
            importances.append(entry['importance'] * 5)  # Scale for marker size
            confidences.append(entry['confidence'])
            descriptions.append(entry['description'][:200] if entry['description'] else 'No description')
            colors.append(color_map.get(entry['event_type'], '#95A5A6'))

        # Create scatter plot for timeline
        fig = go.Figure()

        # Add trace for each event type
        for event_type in set(types):
            type_indices = [i for i, t in enumerate(types) if t == event_type]

            fig.add_trace(go.Scatter(
                x=[dates[i] for i in type_indices],
                y=[1] * len(type_indices),  # All on same horizontal line
                mode='markers',
                name=event_type.capitalize(),
                marker=dict(
                    size=[importances[i] for i in type_indices],
                    color=color_map.get(event_type, '#95A5A6'),
                    line=dict(width=1, color='white'),
                    symbol='circle'
                ),
                text=[titles[i] for i in type_indices],
                customdata=[[
                    confidences[i],
                    importances[i] / 5,
                    descriptions[i]
                ] for i in type_indices],
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Type: ' + event_type + '<br>'
                    'Date: %{x}<br>'
                    'Confidence: %{customdata[0]:.2f}<br>'
                    'Importance: %{customdata[1]}/5<br>'
                    '<extra></extra>'
                )
            ))

        # Update layout
        title_text = f"C64 Knowledge Base Timeline"
        if start_year and end_year:
            title_text += f" ({start_year}-{end_year})"

        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title="",
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                range=[0.5, 1.5]
            ),
            hovermode='closest',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"Timeline visualization saved to {output_file}")
        return str(output_file)

    def visualize_event_network(self, start_year: Optional[int] = None,
                                end_year: Optional[int] = None,
                                output_path: str = "event_network.html") -> str:
        """
        Create network visualization showing relationships between events.

        Events are connected based on:
        - Shared entities (people, companies, technologies)
        - Temporal proximity (events in same year)
        - Event type similarity

        Args:
            start_year: Filter events from this year (optional)
            end_year: Filter events to this year (optional)
            output_path: Path to save HTML file

        Returns:
            Path to saved HTML file

        Examples:
            >>> kb.visualize_event_network(start_year=1980, end_year=1990)
            'event_network.html'
        """
        import plotly.graph_objects as go
        from pathlib import Path
        import networkx as nx

        # Get events
        events = self.search_events_by_date(start_year=start_year, end_year=end_year)

        if not events or len(events) < 2:
            self.logger.warning("Need at least 2 events for network visualization")
            return ""

        # Build network graph
        G = nx.Graph()

        # Add nodes (events)
        for i, event in enumerate(events):
            G.add_node(i,
                      title=event['title'][:50],
                      year=event['year'],
                      type=event['event_type'],
                      confidence=event['confidence'],
                      entities=event['entities'])

        # Add edges based on relationships
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                weight = 0

                # Shared entities
                entities1 = set(event1['entities'])
                entities2 = set(event2['entities'])
                shared = entities1.intersection(entities2)
                if shared:
                    weight += len(shared) * 2

                # Same year
                if event1['year'] == event2['year']:
                    weight += 1

                # Same type
                if event1['event_type'] == event2['event_type']:
                    weight += 0.5

                # Add edge if there's a connection
                if weight > 0:
                    G.add_edge(i, j, weight=weight)

        # Calculate layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        # Prepare edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Prepare node traces by type
        color_map = {
            'release': '#FF6B6B',
            'milestone': '#4ECDC4',
            'innovation': '#95E1D3',
            'cultural': '#FFA07A',
            'update': '#9B59B6'
        }

        node_traces = []
        for event_type in set(node['type'] for _, node in G.nodes(data=True)):
            node_x = []
            node_y = []
            node_text = []
            node_size = []

            for node_id, node_data in G.nodes(data=True):
                if node_data['type'] == event_type:
                    x, y = pos[node_id]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(
                        f"{node_data['title']}<br>"
                        f"Year: {node_data['year']}<br>"
                        f"Type: {node_data['type']}<br>"
                        f"Confidence: {node_data['confidence']:.2f}<br>"
                        f"Connections: {G.degree(node_id)}"
                    )
                    # Node size based on number of connections
                    node_size.append(10 + G.degree(node_id) * 3)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                name=event_type.capitalize(),
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=node_size,
                    color=color_map.get(event_type, '#95A5A6'),
                    line=dict(width=2, color='white')
                )
            )
            node_traces.append(node_trace)

        # Create figure
        fig = go.Figure(data=[edge_trace] + node_traces,
                       layout=go.Layout(
                           title=dict(text='Event Network - C64 Knowledge Base', font=dict(size=16)),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=800
                       ))

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"Event network visualization saved to {output_file}")
        return str(output_file)

    def visualize_event_trends(self, output_path: str = "event_trends.html") -> str:
        """
        Create trend charts showing event distribution over time.

        Generates:
        - Events per year bar chart
        - Events by type over time (stacked area chart)
        - Cumulative events over time

        Args:
            output_path: Path to save HTML file

        Returns:
            Path to saved HTML file

        Examples:
            >>> kb.visualize_event_trends()
            'event_trends.html'
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from pathlib import Path

        cursor = self.db_conn.cursor()

        # Get events by year and type
        events_data = cursor.execute("""
            SELECT year, event_type, COUNT(*) as count
            FROM events
            WHERE year IS NOT NULL
            GROUP BY year, event_type
            ORDER BY year, event_type
        """).fetchall()

        if not events_data:
            self.logger.warning("No events with dates for trend visualization")
            return ""

        # Organize data
        years_set = set()
        type_year_counts = {}

        for year, event_type, count in events_data:
            years_set.add(year)
            if event_type not in type_year_counts:
                type_year_counts[event_type] = {}
            type_year_counts[event_type][year] = count

        years = sorted(years_set)

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Events Per Year',
                'Events by Type Over Time',
                'Cumulative Events'
            ),
            vertical_spacing=0.12,
            row_heights=[0.33, 0.33, 0.33]
        )

        # 1. Events per year (bar chart)
        year_totals = {}
        for year in years:
            year_totals[year] = sum(
                type_year_counts[et].get(year, 0)
                for et in type_year_counts
            )

        fig.add_trace(
            go.Bar(
                x=years,
                y=[year_totals[y] for y in years],
                name='Total Events',
                marker_color='#3498db',
                showlegend=False
            ),
            row=1, col=1
        )

        # 2. Events by type (stacked area chart)
        color_map = {
            'release': '#FF6B6B',
            'milestone': '#4ECDC4',
            'innovation': '#95E1D3',
            'cultural': '#FFA07A',
            'update': '#9B59B6'
        }

        for event_type in sorted(type_year_counts.keys()):
            type_counts = [type_year_counts[event_type].get(y, 0) for y in years]

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=type_counts,
                    name=event_type.capitalize(),
                    mode='lines',
                    stackgroup='one',
                    fillcolor=color_map.get(event_type, '#95A5A6'),
                    line=dict(width=0.5, color=color_map.get(event_type, '#95A5A6'))
                ),
                row=2, col=1
            )

        # 3. Cumulative events
        cumulative = []
        total = 0
        for year in years:
            total += year_totals[year]
            cumulative.append(total)

        fig.add_trace(
            go.Scatter(
                x=years,
                y=cumulative,
                name='Cumulative',
                mode='lines+markers',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8),
                showlegend=False
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Year", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Total Events", row=3, col=1)

        fig.update_layout(
            title_text="Event Trends - C64 Knowledge Base",
            height=1200,
            showlegend=True,
            hovermode='x unified'
        )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"Event trends visualization saved to {output_file}")
        return str(output_file)
