## Hopefully collaborative summary of talks we found interesting...

### Practical Semantic Parsing for Spoken Language Understanding
- Executable semantic parsing for QA and spoken language understanding
    - Could use Slot tagging (i.e., tag the slot that the word will fill) → but here they convert the seq to trees and use Transition based parser
- Exploit high resource domains for low-resource ones:
    - Datasets: Overnight, NLmaps -- low-resource datasets, and different action types have very different number of slots/slot types.
    - Pretraining on high-resource domain helps more consistently than multi-task setting!

