
import copy
from collections import defaultdict
import sys
import numpy as np

try:
    sys.setrecursionlimit(2000)
    print("DEBUG: Recursion limit set to 2000.")
except Exception as e:
    print(f"Warning: Could not set recursion depth limit. {e}")

FLOAT_EPSILON = 1e-6


class SchemaGenerator:
    def __init__(self, episodic_memory, schematic_memory):
        self.episodic_memory = episodic_memory
        self.schematic_memory = schematic_memory
        self.element_mapping = {}
        self.placeholder_store = {}
        self.p_counter = 0
        self.d_counter = 0
        self.event_category_cache = {}

    def _get_event_connections(self, instance):
        # ... (代码不变) ...
        event_connections = defaultdict(list)
        event_map = {}
        for e in instance['events']:
            seq = e.get('seq')
            cat = e.get('category')
            if seq is not None and cat != 'padding':
                if (seq, cat) not in event_map:
                    event_map[(seq, cat)] = e

        for conn in instance.get("connections", []):
            s1 = conn.get("event1_seq")
            s2 = conn.get("event2_seq")
            common = conn.get("common_elements")
            if s1 is None or s2 is None or not common:
                continue
            e1, e2, cat1, cat2 = None, None, None, None
            possible_cats_s1 = [cat for (sq, cat), evt in event_map.items() if sq == s1]
            possible_cats_s2 = [cat for (sq, cat), evt in event_map.items() if sq == s2]
            for c in possible_cats_s1:
                if c in ["ball_motion", "arm_motion", "ball_speed", "arm_speed"]:
                    e1 = event_map.get((s1, c))
                    cat1 = c
                    break
            for c in possible_cats_s2:
                if c in ["ball_motion", "arm_motion", "ball_speed", "arm_speed"]:
                    e2 = event_map.get((s2, c))
                    cat2 = c
                    break
            if not e1 or not e2 or not cat1 or not cat2:
                continue
            key1 = (s1, cat1)
            event_connections[key1].append({
                "connected_seq": s2,
                "connected_cat": cat2,
                "common": common
            })
            key2 = (s2, cat2)
            event_connections[key2].append({
                "connected_seq": s1,
                "connected_cat": cat1,
                "common": common
            })
        return event_connections

    def generate_schema(self, instance1, instance2):
        print("DEBUG: Starting schema generation (sequence-based, category-aware, no_tags)...")
        if not instance1 or not instance2 or not instance1.get("events") or not instance2.get("events"):
            print("ERROR: Invalid input instances for advanced generation.")
            return
        self.element_mapping = {}
        self.placeholder_store = {}
        self.p_counter = 0
        self.d_counter = 0
        self.event_category_cache.clear()
        print(f"DEBUG: Reset element mapping, counters, and category cache.")
        generalized_events = self._generalize_events_by_sequence(instance1, instance2)
        print(f"DEBUG: Step 1 (Generalization) produced {len(generalized_events)} events.")
        if not generalized_events:
            print("WARN: Generalization step resulted in no events. Aborting.")
            return
        final_advanced_events = self._inject_abstract_identifiers(generalized_events, instance1, instance2)
        print(f"DEBUG: Step 2 (Identifier Injection) resulted in {len(final_advanced_events)} events.")
        schema_data = {"events": final_advanced_events,
                                  "connections": []}  # Connections are not generalized yet
        if schema_data["events"]:
            print(f"DEBUG: Successfully generated schema (no_tags).")
            self.schematic_memory.add_schema(schema_data)
            return schema_data
        else:
            print("DEBUG: Advanced instance creation resulted in no events after injection step (no_tags).")
            return

    def _get_event_map(self, events):
        event_map = defaultdict(list)
        for event in events:
            seq = event.get('seq')
            if seq is not None:
                event_map[seq].append(event)
        return event_map

    # --- MODIFIED _generalize_events_by_sequence ---
    def _generalize_events_by_sequence(self, instance1, instance2):

        events1 = instance1["events"]
        events2 = instance2["events"]

        map1 = self._get_event_map(events1)
        map2 = self._get_event_map(events2)

        max_seq1 = max(map1.keys()) if map1 else 0
        max_seq2 = max(map2.keys()) if map2 else 0
        max_seq = max(max_seq1, max_seq2)

        generalized_events = []

        for s in range(1, max_seq + 1):
            events_at_s1 = [e for e in map1.get(s, []) if e.get('category') != 'padding']
            events_at_s2_mutable = [e for e in map2.get(s, []) if e.get('category') != 'padding']

            all_padding1 = not events_at_s1
            all_padding2 = not events_at_s2_mutable

            if all_padding1 and all_padding2:
                continue

            if not events_at_s1 and not events_at_s2_mutable:
                generalized_events.append(
                    {"seq": s, "category": "padding", "content": {"type": "empty_sequence"}})
                continue

            for e1 in events_at_s1:
                e2_match = None
                match_index = -1
                for i, e2_candidate in enumerate(events_at_s2_mutable):
                    if e1['category'] == e2_candidate['category']:
                        e2_match = e2_candidate
                        match_index = i
                        break

                cat = e1['category']
                adv_event = None

                if e2_match:
                    events_at_s2_mutable.pop(match_index)

                    content1 = copy.deepcopy(e1.get('content', {}))
                    content2 = copy.deepcopy(e2_match.get('content', {}))


                    content1.pop('tags', None)
                    content2.pop('tags', None)


                    if cat == 'command':
                        content1.pop('rotation', None)
                        content2.pop('rotation', None)

                    if content1 == content2:
                        adv_event = {"seq": s, "category": cat, "content": copy.deepcopy(content1)}
                    else:
                        generalized_content = {}
                        all_keys = set(content1.keys()) | set(content2.keys())
                        for k in all_keys:
                            v1 = content1.get(k)
                            v2 = content2.get(k)
                            if k in content1 and k in content2:
                                generalized_content[k] = v1 if v1 == v2 else "ANY_PLACEHOLDER"
                            else:
                                generalized_content[k] = "ANY_PLACEHOLDER"
                        adv_event = {"seq": s, "category": cat, "content": generalized_content}
                else:

                    adv_event = {"seq": s, "category": "padding",
                                 "content": {"type": f"missing_event_in_instance2_{cat}"}}

                if adv_event:
                    generalized_events.append(adv_event)


            for e2_remaining in events_at_s2_mutable:
                cat = e2_remaining['category']
                adv_event = {"seq": s, "category": "padding", "content": {"type": f"missing_event_in_instance1_{cat}"}}
                generalized_events.append(adv_event)


        final_generalized_events = []
        last_event_was_padding = False
        for event in generalized_events:
            is_padding = event.get("category") == "padding"
            if is_padding:
                if not last_event_was_padding:
                    final_generalized_events.append(event)
                    last_event_was_padding = True
            else:
                final_generalized_events.append(event)
                last_event_was_padding = False

        return final_generalized_events

    def _get_placeholder(self, key_prefix, val1, val2):

        try:
            def make_hashable(v):
                if isinstance(v, list): return tuple(make_hashable(i) for i in v)
                if isinstance(v, dict): return tuple(sorted((k, make_hashable(v[k])) for k in v))
                if isinstance(v, (np.ndarray, np.generic)): return tuple(v.tolist())
                try:
                    hash(v);
                    return v
                except TypeError:
                    return str(v)

            h_val1 = make_hashable(val1)
            h_val2 = make_hashable(val2)
            hash(h_val1);
            hash(h_val2)
        except TypeError:
            print(
                f"WARN: Values {(val1, type(val1))}, {(val2, type(val2))} not hashable even after conversion. Returning 'ANY_PLACEHOLDER'.")
            return "ANY_PLACEHOLDER"
        if h_val1 == h_val2: return val1
        pair_tuple = (str(h_val1), str(h_val2))  # Ensure tuple elements are strings for consistent hashing
        pair_key = f"{key_prefix}_{pair_tuple}"
        if pair_key in self.element_mapping: return self.element_mapping[pair_key]
        if key_prefix == "P":
            self.p_counter += 1;
            placeholder = f"P{self.p_counter}"
        elif key_prefix == "D":
            self.d_counter += 1;
            placeholder = f"D{self.d_counter}"
        else:
            prefix_upper = key_prefix.upper()
            count = sum(1 for k in self.element_mapping if k.startswith(prefix_upper)) + 1
            placeholder = f"{prefix_upper}{count}"
        self.element_mapping[pair_key] = placeholder
        self.placeholder_store[placeholder] = (val1, val2)
        # print(f"DEBUG: Created placeholder {placeholder} for key {pair_key} representing ({val1}, {val2})") # Kept for debugging
        return placeholder

    def _inject_abstract_identifiers(self, generalized_events, instance1, instance2):
        print("DEBUG: Injecting abstract identifiers (v2 - direct modification)...")
        advanced_event_map = {}
        for event in generalized_events:
            seq = event.get('seq')
            cat = event.get('category')
            if seq is not None and cat != 'padding':
                advanced_event_map[(seq, cat)] = event
        event_map1 = {(e['seq'], e['category']): e for e in instance1['events'] if e['category'] != 'padding'}
        event_map2 = {(e['seq'], e['category']): e for e in instance2['events'] if e['category'] != 'padding'}
        connections_map2 = {}
        for conn2 in instance2.get("connections", []):
            s1_2, s2_2, common2 = conn2.get("event1_seq"), conn2.get("event2_seq"), conn2.get("common_elements")
            if s1_2 is None or s2_2 is None or not common2: continue
            cat1_2 = self._find_event_category_for_connection(event_map2, s1_2, common2)
            cat2_2 = self._find_event_category_for_connection(event_map2, s2_2, common2)
            if not cat1_2 or not cat2_2: continue
            conn_key2 = self._normalize_connection_key(s1_2, s2_2, cat1_2, cat2_2)
            connections_map2[conn_key2] = {"s1": s1_2, "s2": s2_2, "cat1": cat1_2, "cat2": cat2_2, "common": common2}
        # print(f"DEBUG: Processing {len(instance1.get('connections', []))} connections from instance1...")
        connections_processed_count = 0
        placeholder_applications = 0
        for conn1 in instance1.get("connections", []):
            connections_processed_count += 1
            s1_1, s2_1, common1 = conn1.get("event1_seq"), conn1.get("event2_seq"), conn1.get("common_elements")
            if s1_1 is None or s2_1 is None or not common1: continue
            cat1_1 = self._find_event_category_for_connection(event_map1, s1_1, common1)
            cat2_1 = self._find_event_category_for_connection(event_map1, s2_1, common1)
            if not cat1_1 or not cat2_1: continue
            conn_key1 = self._normalize_connection_key(s1_1, s2_1, cat1_1, cat2_1)
            matched_conn2_info = connections_map2.get(conn_key1)
            if matched_conn2_info:
                s1_2, s2_2, cat1_2, cat2_2, common2 = matched_conn2_info["s1"], matched_conn2_info["s2"], \
                    matched_conn2_info["cat1"], matched_conn2_info["cat2"], matched_conn2_info["common"]
                common_keys_intersect = set(common1.keys()) & set(common2.keys())
                for c_key in common_keys_intersect:
                    placeholder_prefix = None
                    if c_key == "current_pos":
                        placeholder_prefix = "P"
                    elif c_key == "new_direction":
                        placeholder_prefix = "D"
                    if placeholder_prefix:
                        val1_e1 = self._get_original_value(instance1, s1_1, cat1_1, c_key)
                        val2_e1 = self._get_original_value(instance2, s1_2, cat1_2, c_key)
                        if val1_e1 is not None and val2_e1 is not None and val1_e1 != val2_e1:
                            adv_event1 = advanced_event_map.get((s1_1, cat1_1))
                            if adv_event1 and isinstance(adv_event1.get('content'), dict):
                                placeholder_e1 = self._get_placeholder(placeholder_prefix, val1_e1, val2_e1)
                                current_adv_val = adv_event1['content'].get(c_key)
                                if current_adv_val != placeholder_e1:
                                    # print(
                                    #     f"DEBUG: Applying placeholder '{placeholder_e1}' to adv_event ({s1_1}, {cat1_1}) key '{c_key}'. Orig vals: ({val1_e1} vs {val2_e1}). Old adv val: '{current_adv_val}'")
                                    adv_event1['content'][c_key] = placeholder_e1
                                    placeholder_applications += 1
                        val1_e2 = self._get_original_value(instance1, s2_1, cat2_1, c_key)
                        val2_e2 = self._get_original_value(instance2, s2_2, cat2_2, c_key)
                        if val1_e2 is not None and val2_e2 is not None and val1_e2 != val2_e2:
                            adv_event2 = advanced_event_map.get(
                                (s2_1, cat2_1))  # Should be s2_1, cat2_1 if mapping to instance1's timeline
                            if adv_event2 and isinstance(adv_event2.get('content'), dict):
                                placeholder_e2 = self._get_placeholder(placeholder_prefix, val1_e2, val2_e2)
                                current_adv_val = adv_event2['content'].get(c_key)
                                if current_adv_val != placeholder_e2:
                                    # print(
                                    #     f"DEBUG: Applying placeholder '{placeholder_e2}' to adv_event ({s2_1}, {cat2_1}) key '{c_key}'. Orig vals: ({val1_e2} vs {val2_e2}). Old adv val: '{current_adv_val}'")
                                    adv_event2['content'][c_key] = placeholder_e2
                                    placeholder_applications += 1
        # print(
        #     f"DEBUG: Finished injection phase. Processed {connections_processed_count} instance1 connections. Applied/verified {placeholder_applications} placeholders.")
        return generalized_events

    def _find_event_category_for_connection(self, event_map, seq, common_elements):
        possible_cats = []
        if 'new_direction' in common_elements or 'current_pos' in common_elements:
            possible_cats.extend(['ball_motion', 'arm_motion'])
        if 'change' in common_elements:
            possible_cats.extend(['ball_speed', 'arm_speed'])
        for cat in possible_cats:
            if (seq, cat) in event_map: return cat
        # Fallback if specific common elements don't match typical categories
        for (sq, cat), evt in event_map.items():
            if sq == seq and cat in ["ball_motion", "arm_motion", "ball_speed", "arm_speed"]: return cat
        return None

    def _normalize_connection_key(self, s1, s2, cat1, cat2):
        if s1 < s2:
            return (s1, s2, cat1, cat2)
        elif s2 < s1:
            return (s2, s1, cat2, cat1)
        else:  # s1 == s2
            if cat1 <= cat2:  # Lexicographical comparison for categories
                return (s1, s2, cat1, cat2)
            else:
                return (s2, s1, cat2, cat1)

    def _get_original_value(self, instance, seq, category, value_key):
        event_map = {(e['seq'], e['category']): e for e in instance['events'] if e['category'] != 'padding'}
        target_event = event_map.get((seq, category))
        if target_event and isinstance(target_event.get('content'), dict):
            return target_event['content'].get(value_key)
        return None

# --- END OF FILE schema_generator (21)_no_tags.py ---