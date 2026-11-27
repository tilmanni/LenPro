from collections.abc import Sequence
from typing import Any

from lenpro.measures.aggregate_measures import AggregateMeasure
from lenpro.measures.entry_level_measures import EntryLevelMeasure
from lenpro.measures.global_measures import GlobalMeasure
from lenpro.protocols.ldp_protocol import LDPProtocol, Stream


Dataset = Sequence[Stream]

# If i want to sample from a discrete laplace distribution: 
# https://mathoverflow.net/questions/213221/what-is-a-two-sided-geometric-distribution#345492


class Aggregator:

    def __init__(self) -> None:
        self.protocols: list[tuple[LDPProtocol, tuple[str, str]]] = []
        self.entry_level_measures: list[EntryLevelMeasure] = []
        self.aggregate_measures: list[AggregateMeasure] = []
        self.global_measures: list[GlobalMeasure] = []
    
    def register_protocol(self, ldp_protocol: LDPProtocol, protocol_meta_data: tuple[str, str] | None = None) -> None: 
        # Have to also register protocol variant label / id for later plotting
        if protocol_meta_data is None:
            name = type(ldp_protocol).__name__
            protocol_meta_data = (name, name)
        self.protocols.append((ldp_protocol, protocol_meta_data))

    def register_entry_level_measure(self, entry_level_measure: EntryLevelMeasure) -> None:
        self.entry_level_measures.append(entry_level_measure)

    def register_aggregate_measure(self, aggregate_measure: AggregateMeasure) -> None:
        self.aggregate_measures.append(aggregate_measure)

    def register_global_measure(self, global_measure: GlobalMeasure) -> None:
        self.global_measures.append(global_measure) 

    def run_simulation(self, input_stream: Stream) -> list[Stream]:
        return [protocol[0].process_stream(input_stream) for protocol in self.protocols]    

    def measure_run(self, input_streams: Dataset) -> list[dict[str, Any]]:
        results = []
        for protocol in self.protocols:

            protocol_outputs = [protocol[0].process_stream(stream) for stream in input_streams]

            entry_level_results = []
            for measure in self.entry_level_measures:
                values = [
                    measure.process_data(original_data=original, perturbed_data=perturbed, protocol=protocol[0]) for original, perturbed in zip(input_streams, protocol_outputs)
                ]
                entry_level_results.append({
                    "measure": measure.to_dict(),
                    "values": values,
                })

            aggregate_results = []
            for aggregate_measure in self.aggregate_measures:
                for entry in entry_level_results:
                    aggregate_results.append({
                        "measure": aggregate_measure.to_dict(),
                        "entry_level_measure": entry["measure"],
                        "value": aggregate_measure.process_measures(entry["values"]),
                    })

            global_results = []
            for global_measure in self.global_measures:
                global_results.append({
                    "measure": global_measure.to_dict(),
                    "value": global_measure.process_dataset(original_dataset=input_streams, perturbed_dataset=protocol_outputs, protocol=protocol[0]),
                })

            # Index by protocol variant as well? 
            results.append({
                "protocol_id": protocol[1][0],
                "protocol_label": protocol[1][1],
                "protocol": protocol[0].to_dict(),
                # For now, do not store entry level results.
                #"entry_level": entry_level_results,
                "aggregate": aggregate_results,
                "global": global_results,
            })
        return results