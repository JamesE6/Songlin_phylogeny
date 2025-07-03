#!/usr/bin/env python

"""
Build a NEXUS file from raw data, suited for BEAUTi.

The script intentionally uses only standard Python dependencies, to make
the results easier and faster to reproduce.
"""

# Import Python standard libraries
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import csv
import logging
import re
import os
import unidecode
from typing import *

BASE_PATH = Path(__file__).parent


def slug_language(text: str) -> str:
    """
    Slug text for language labels.

    This function is used to normalize language names, so that they
    can be used as labels in the NEXUS file.

    Parameters
    ----------
    text : str
        The text to slug.

    Returns
    -------
    slug : str
        The slugged text.
    """

    # Characters to be dropped and replaced
    dropchars = "()[]'"
    spacechars = "-_"

    # Drop unwanted characters
    text = "".join(c for c in text if c not in dropchars)

    # Replace space characters with actual space
    text = "".join(" " if c in spacechars else c for c in text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text.strip())

    # Replace spaces with underscore
    text = text.replace(" ", "_")

    # Decode to ASCII
    return unidecode.unidecode(text)


def slug_concept(text: str) -> str:
    """
    Slug text for concept labels.

    This function is used to normalize concept names, so that they
    can be used as labels in the NEXUS file.

    Parameters
    ----------
    text : str
        The text to slug.

    Returns
    -------
    slug : str
        The slugged text.
    """

    text = text.lower()
    for dropchar in "-_ ()[]'":
        text = text.replace(dropchar, "")

    return unidecode.unidecode(text)


def read_raw_data(args: argparse.Namespace) -> List[Dict[str, str]]:
    """
    Read raw data from a csv file using the `csv` standard library.

    This function also handles removing EDICTOR/LingPy metadata, if any, and checks
    if the requested column names are in the data.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the script. It should include the input csv file path
        and the column names for language, concept, and value.

    Returns
    -------
    data : List[Dict[str, str]]
        The data as a list of dictionaries where each dictionary corresponds to a row in the csv file.

    Raises
    ------
    ValueError
        If one or more requested columns are missing from the csv file.
    FileNotFoundError
        If the input file cannot be found.
    """
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file {args.input} not found.")

    # Determine delimiter based on file extension
    file_ext = os.path.splitext(args.input)[1].lower()
    with open(args.input, encoding="utf-8") as handler:
        if file_ext == ".tsv":
            reader = csv.DictReader(handler, delimiter="\t")
        else:
            sample = handler.read(1024)
            handler.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(handler, dialect=dialect)
        data = list(reader)

    # Remove EDICTOR metadata, if any
    first_col = list(data[0])[0]
    data = [row for row in data if not row[first_col].startswith("#@")]

    # Check if the requested column names are in the data
    requested_columns = [args.language, args.concept, args.value]
    missing_columns = [col for col in requested_columns if col not in data[0]]

    if missing_columns:
        raise ValueError(f"One or more requested columns missing: {missing_columns}")

    return data


def get_triplets(data, args: argparse.Namespace) -> List[Dict[str, str]]:
    """
    Extract language/concept/cognate triplets from data.

    The function also takes care of normalizing/slugging names,
    etc. It also filters out entries with empty values. The
    function also takes care of filtering out constant features
    (i.e., features that have the same value for all languages).

    Parameters
    ----------
    data : List[Dict[str, str]]
        The data as a list of dictionaries where each dictionary corresponds to a row in the csv file.
    args : argparse.Namespace
        The arguments passed to the script. It should include the column names for language, concept, and value.

    Returns
    -------
    entries : List[Dict[str, str]]
        The data as a list of dictionaries where each dictionary corresponds to a row in the csv file.
    """

    def process_value(value: str) -> str:
        return value.strip() or "?"

    # Prepare initial entries
    entries = [
        {
            "Language": slug_language(entry[args.language]),
            "Concept": slug_concept(entry[args.concept]),
            "Cognate": process_value(entry[args.value]),
        }
        for entry in data
        if process_value(entry[args.value]) != "?"
    ]

    # Make sure there are no constant features, after all the filtering
    num_values = defaultdict(set)
    for entry in entries:
        num_values[entry["Concept"]].add(entry["Cognate"])
    num_values = {concept: len(cognates) for concept, cognates in num_values.items()}
    singletons = [concept for concept, num in num_values.items() if num == 1]
    if singletons:
        logging.info(
            f"There are {len(singletons)} singletons in concepts, filtering from {len(entries)} entries..."
        )
        entries = [entry for entry in entries if entry["Concept"] not in singletons]
        logging.info(f"Down to {len(entries)} entries...")

    return entries


def collect_info(
    entries: List[Dict[str, str]]
) -> Tuple[Set[str], Set[str], Dict[Tuple[str, str], Set[str]]]:
    """
    Collects information on a set of entries.

    The information includes list of languages, concepts, and a lookup. This is
    used multiple times both by the filtering function and the output ones.

    Parameters
    ----------
    entries : List[Dict[str, str]]
        The data as a list of dictionaries where each dictionary corresponds to a row in the csv file.

    Returns
    -------
    languages : Set[str]
        The set of languages.
    concepts : Set[str]
        The set of concepts.
    lookup : Dict[Tuple[str, str], Set[str]]
        A lookup table from (language, concept) to the observed cognates.
    """

    # Collect all languages, concepts, and cognates per concept; `cognates` is
    # first collected a list, so that we can derive (by means of a Counter)
    # the states in inverted frequency order. We also build a lookup table
    # from (language, concept) to the observed cognates (which can be more
    # than one, for synonyms).
    languages = set()
    concepts = set()
    cognates = defaultdict(list)
    lookup = defaultdict(set)
    for entry in entries:
        language, concept, cognate = (
            entry["Language"],
            entry["Concept"],
            entry["Cognate"],
        )

        languages.add(language)
        concepts.add(concept)
        if cognate not in ["?", "0"]:  # TODO: uralex codes, confirm they are general
            cognates[concept].append(cognate)
            lookup[language, concept].add(cognate)

    cognates = {
        concept: [item[0] for item in Counter(coglist).most_common()]
        for concept, coglist in cognates.items()
    }

    return languages, concepts, cognates, lookup


# TODO: allow to build multistate, non-binary in the future (use panphylo?)
def build_nexus(entries, args):
    """
    Build a NEXUS source from a list of entries.

    Note that this assumes the contents have already been cleaned
    and filtered.
    """

    # Collect all languages, concepts, and cognates per concept; `cognates` is
    # first collected a list, so that we can derive (by means of a Counter)
    # the states in inverted frequency order. We also build a lookup table
    # from (language, concept) to the observed cognates (which can be more
    # than one, for synonyms).
    languages = set()
    concepts = set()
    cognates = defaultdict(list)
    lookup = defaultdict(set)
    for entry in entries:
        language, concept, cognate = (
            entry["Language"],
            entry["Concept"],
            entry["Cognate"],
        )

        languages.add(language)
        concepts.add(concept)
        if cognate not in ["?", "0"]:  # TODO: uralex codes, confirm they are general
            cognates[concept].append(cognate)
            lookup[(language, concept)].add(cognate)

    cognates = {
        concept: [item[0] for item in Counter(coglist).most_common()]
        for concept, coglist in cognates.items()
    }

    # Iterate over the lookup to build the matrix and charstates
    matrix = defaultdict(str)
    charstatelabels = []
    assumptions = []
    prev_idx = 0
    for concept in sorted(concepts):
        # Extend charstate
        charstatelabels.append(f"{concept}_0ascertainment")
        for cognate in cognates[concept]:
            charstatelabels.append(f"{concept}_{cognate}")
        assumptions.append(
            [concept, prev_idx + 1, prev_idx + 1 + len(cognates[concept])]
        )
        prev_idx = assumptions[-1][-1]

        # Extend matrix
        for language in languages:
            # Build an observed vector, replacing it by missing data if not cognate is
            # observed for the pair
            observed = [
                "1" if cognate in lookup[language, concept] else "0"
                for cognate in cognates[concept]
            ]
            if all([state == "0" for state in observed]):
                if args.zero_ascert:
                    matrix[language] += "0"  # ascertainment
                else:
                    matrix[language] += "?"  # ascertainment
                matrix[language] += "?" * len(observed)
            else:
                matrix[language] += "0"  # ascertainment
                matrix[language] += "".join(observed)

    # Read template and perform replacements
    with open(BASE_PATH / "etc" / "nexus.template", encoding="utf-8") as handler:
        nexus = handler.read()

    nexus = nexus.replace("$$NTAXA$$", str(len(languages)))
    nexus = nexus.replace("$$NCHAR$$", str(len(charstatelabels)))

    charstatelabels_str = ",\n".join(
        [
            "        %i %s" % (idx + 1, charstatelabel)
            for idx, charstatelabel in enumerate(charstatelabels)
        ]
    )
    nexus = nexus.replace("$$CHARSTATELABELS$$", charstatelabels_str)

    max_taxon_length = max([len(taxon) for taxon in languages])
    matrix_str = "\n".join(
        [
            "%s %s" % (language.ljust(max_taxon_length), vector)
            for language, vector in sorted(matrix.items())
        ]
    )
    nexus = nexus.replace("$$MATRIX$$", matrix_str)

    assumptions_str = "\n".join(
        [
            "    CHARSET %s = %i-%i;" % (character, start, end)
            for character, start, end in assumptions
        ]
    )
    nexus = nexus.replace("$$ASSUMPTIONS$$", assumptions_str)

    return nexus


def filter_data(entries, args):
    """
    Performs all the necessary filtering in terms of languages and concepts.
    """

    # Filter out languages without enough coverage
    languages, concepts, cognates, lookup = collect_info(entries)
    coverage = {
        language: len([concept for concept in concepts if lookup[language, concept]])
        / len(concepts)
        for language in languages
    }

    prev_size = len(entries)
    entries = [
        entry for entry in entries if coverage[entry["Language"]] > args.coverage
    ]
    logging.info(f"Filtering low coverage ({prev_size} -> {len(entries)} entries).")

    # Filter out concepts without enough frequency
    languages, concepts, cognates, lookup = collect_info(entries)
    frequency = {
        concept: len(
            set([language for language in languages if lookup[language, concept]])
        )
        / len(languages)
        for concept in concepts
    }

    prev_size = len(entries)
    entries = [
        entry for entry in entries if frequency[entry["Concept"]] > args.frequency
    ]
    logging.info(f"Filtering low frequency ({prev_size} -> {len(entries)} entries).")

    # Filter out concepts with a single observed cognate
    # TODO: implement filtering that keeps missing data?
    languages, concepts, cognates, lookup = collect_info(entries)
    prev_len = len(entries)
    entries = [entry for entry in entries if len(cognates[entry["Concept"]]) > 1]
    logging.info(
        f"Filtering out constant concepts 2nd time ({prev_len} -> {len(entries)} entries)."
    )

    return entries


def get_statistics(data, args):
    """
    Gather statistics about the data and write them to CSV files.
    """

    # Define the statistics dictionaries
    lang_stats = defaultdict(
        lambda: {
            "total_entries": 0,
            "concepts_cognates": defaultdict(set),
            "concepts_with_cognates": set(),
        }
    )
    concept_stats = defaultdict(lambda: {"unique_cognates": set(), "coverage": set()})
    concepts_total = set()
    languages_total = set()

    # Collect statistics
    for entry in data:
        doculect = entry[args.language]
        concept = entry[args.concept]
        cogid = entry[args.value]

        # Do not count entries without a cognate
        if not cogid:
            continue

        lang_stats[doculect]["total_entries"] += 1
        lang_stats[doculect]["concepts_cognates"][concept].add(cogid)
        lang_stats[doculect]["concepts_with_cognates"].add(concept)

        concept_stats[concept]["unique_cognates"].add(cogid)
        concept_stats[concept]["coverage"].add(doculect)

        concepts_total.add(concept)
        languages_total.add(doculect)

    # Write statistics to CSV
    with open(
        BASE_PATH / "data" / f"language_statistics.{args.output}.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Language",
                "Total Entries",
                "Concepts with Cognates (Absolute)",
                "Concepts with Cognates (Relative)",
            ]
        )

        for doculect in sorted(lang_stats.keys()):
            writer.writerow(
                [
                    doculect,
                    lang_stats[doculect]["total_entries"],
                    len(lang_stats[doculect]["concepts_with_cognates"]),
                    len(lang_stats[doculect]["concepts_with_cognates"])
                    / len(concepts_total),
                ]
            )

    with open(
        BASE_PATH / "data" / f"concept_statistics.{args.output}.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Concept", "Unique Cognates", "Coverage (Absolute)", "Coverage (Relative)"]
        )

        for concept in sorted(concept_stats.keys()):
            writer.writerow(
                [
                    concept,
                    len(concept_stats[concept]["unique_cognates"]),
                    len(concept_stats[concept]["coverage"]),
                    len(concept_stats[concept]["coverage"]) / len(languages_total),
                ]
            )

    for doculect in sorted(lang_stats.keys()):
        with open(
            BASE_PATH / "data" / f"concepts_{doculect.lower()}.{args.output}.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Concept", "Cognates"])

            for concept in sorted(lang_stats[doculect]["concepts_cognates"].keys()):
                writer.writerow(
                    [concept, len(lang_stats[doculect]["concepts_cognates"][concept])]
                )


def main(args):
    # Read raw data
    logging.info(f"Reading entries from raw source.")
    data = read_raw_data(args)
    logging.info(f"Read {len(data)} entries from raw source.")

    # Doculect renaming and filtering
    if args.doculects:
        # Read data
        with open(args.doculects, "r", encoding="utf-8") as f:
            lang_map = {}
            for row in f:
                if row.strip():
                    if ":" in row:
                        lang_map[row.split(":")[0]] = row.split(":")[1].strip()
                    else:
                        lang_map[row.strip()] = row.strip()

        filtered_entries = []
        for entry in data:
            if entry[args.language] in lang_map:
                entry[args.language] = lang_map[entry[args.language]]
                filtered_entries.append(entry)

        prev_size = len(data)
        data = filtered_entries
        logging.info(f"Filtering doculects ({prev_size} -> {len(data)} entries).")

    # Parameter renaming and filtering
    if args.parameters:
        # Read data
        with open(args.parameters, "r", encoding="utf-8") as f:
            concept_map = {}
            for row in f:
                if row.strip():
                    if ":" in row:
                        concept_map[row.split(":")[0]] = row.split(":")[1].strip()
                    else:
                        concept_map[row.strip()] = row.strip()

        filtered_entries = []
        for entry in data:
            if entry[args.concept] in concept_map:
                entry[args.concept] = concept_map[entry[args.concept]]
                filtered_entries.append(entry)

        prev_size = len(data)
        data = filtered_entries
        logging.info(f"Filtering concepts ({prev_size} -> {len(data)} entries).")

    # Collect statistics
    get_statistics(data, args)

    # Extract triplets
    logging.info(f"Extracting triplet entries from the data and filtering.")
    entries = get_triplets(data, args)
    entries = filter_data(entries, args)

    # Build and write NEXUS source
    nexus = build_nexus(entries, args)
    nexus_file = BASE_PATH / "data" / f"sinotibetan.{args.output}.nex"
    if Path(nexus_file).is_file() and not args.overwrite:
        raise ValueError(
            f"Output file `{args.output}` already exists. Use `--overwrite` if you are sure."
        )
    else:
        logging.info(f"Writing output to `{args.output}`.")
        with open(nexus_file, "w", encoding="utf-8") as handler:
            handler.write(nexus)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare Nexus file.")
    parser.add_argument("input", type=str, help="Input tabular file.")
    parser.add_argument("output", type=str, help="The name of the output dataset.")
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        help="Language column (default: `DOCULECT`)",
        default="DOCULECT",
    )
    parser.add_argument(
        "-c",
        "--concept",
        type=str,
        help="Concept column (default: `CONCEPT`)",
        default="CONCEPT",
    )
    parser.add_argument(
        "-v",
        "--value",
        type=str,
        help="Value/cognate column (default: `COGID`)",
        default="COGID",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file (default: preserve it).",
    )
    parser.add_argument(
        "-z",
        "--zero_ascert",
        action="store_true",
        help="Use zero in ascertainment for missing data (default: use question marks).",
    )
    parser.add_argument(
        "-d",
        "--doculects",
        type=str,
        help="File with the list of language labels to keep when filtering.",
    )
    parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        help="File with the list of concept labels to keep when filtering.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        help="Minimum concept coverage for including a language (default 0.0, to include all).",
        default=0.0,
    )
    parser.add_argument(
        "--frequency",
        type=float,
        help="Minimum language frequency for including a concept (default 0.0, use 0.0 to include all).",
        default=0.0,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse command-line, set logging, and begin; arguments to `main()`
    ARGS = parse_args()
    logging.basicConfig(level=logging.INFO)
    main(ARGS)
