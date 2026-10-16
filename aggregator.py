from collections import Counter
from protocols import ldp_protocol
from measures import aggregate_measures, entry_level_measures, global_measures


# If i want to sample from a discrete laplace distribution: 
# https://mathoverflow.net/questions/213221/what-is-a-two-sided-geometric-distribution#345492


class Aggregator:

    def __init__(self) -> None:
        self.protocols = []
        self.entry_level_measures = []
        self.aggregate_measures = []
        self.global_measures = []

    def register_protocol(self, ldp_protocol: ldp_protocol.LDPProtocol) -> None:
        self.protocols.append(ldp_protocol)

    def register_entry_level_measure(self, entry_level_measure: entry_level_measures.EntryLevelMeasure) -> None:
        self.entry_level_measures.append(entry_level_measure)

    def register_aggregate_measure(self, aggregate_measure: aggregate_measures.AggregrateMeasure) -> None:
        self.aggregate_measures.append(aggregate_measure)

    def register_global_measure(self, global_measure: global_measures.GlobalMeasure) -> None:
        self.global_measures.append(global_measure) 


    def run_simulation(self, input):
        protocol_outputs = []
        for protocol in self.protocols:
            protocol_output = []
            for event in input:
                protocol_output.append(protocol.process_event(event))
            protocol_outputs.append(protocol_output)
        return protocol_outputs

    # Each run could signify one user, as independent.
    def get_full_count_distribution(self, input, runs=100):
        event_counts = []
        for protocol in self.protocols:
            event_counts.append(Counter())

        for i in range (0, runs):
            protocol_outputs = self.run_simulation(input=input)
            amount_of_protocols = len(self.protocols)
            for j in range(0, amount_of_protocols):
                # Run local measures
                number_of_events = sum(x is not None for x in protocol_outputs[j])
                event_counts[j][number_of_events] += 1
        return event_counts
    

    def measure_run(self, inputs):
        results = []
        for protocol in self.protocols:

            protocol_outputs = [protocol.process_stream(stream) for stream in inputs]

            entry_level_results = []
            for measure in self.entry_level_measures:
                values = [
                    measure.process_data(original, perturbed) for original, perturbed in zip(inputs, protocol_outputs)
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
                    "value": global_measure.process_dataset(original_dataset=inputs, perturbed_dataset=protocol_outputs),
                })

            results.append({
                "protocol": protocol.to_dict(),
                "entry_level": entry_level_results,
                "aggregate": aggregate_results,
                "global": global_results,
            })
        return results