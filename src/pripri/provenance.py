provenance_roots = {}
provenance_tag_counts = {}

class ProvenanceEntity:
    def __init__(self, parent, tag, children_type):
        self.parent = parent
        self.tag = tag
        self.children_type = children_type
        self.privacy_budget_local = 0
        self.privacy_budget = 0
        self.children = []

    def update_privacy_budget(self):
        if self.children_type == "inclusive":
            self.privacy_budget = self.privacy_budget_local + sum([c.privacy_budget for c in self.children])

            if self.parent is not None:
                self.parent.update_privacy_budget()

        elif self.children_type == "exclusive":
            self.privacy_budget = self.privacy_budget_local + max([0] + [c.privacy_budget for c in self.children])

            if self.parent is not None:
                self.parent.update_privacy_budget()

        else:
            raise RuntimeError

    def accumulate_privacy_budget(self, privacy_budget):
        assert self.children_type == "inclusive"

        self.privacy_budget_local += privacy_budget

        self.update_privacy_budget()

    def add_child(self, children_type):
        pe = ProvenanceEntity(self, self.tag, children_type)
        self.children.append(pe)
        return pe

    def new_tag(self):
        global provenance_tag_counts

        name, count = self.tag

        assert name in provenance_tag_counts
        provenance_tag_counts[name] += 1

        self.tag = (name, provenance_tag_counts[name])

def new_provenance_root(name):
    global provenance_roots
    global provenance_tag_counts

    if name in provenance_roots:
        raise ValueError(f"Name '{name}' already exists")

    initial_tag_count = 0
    provenance_tag_counts[name] = initial_tag_count

    pe = ProvenanceEntity(None, (name, initial_tag_count), "inclusive")
    provenance_roots[name] = pe

    return pe

def get_privacy_budget(name):
    global provenance_roots

    if name not in provenance_roots:
        raise ValueError(f"Name '{name}' does not exist")

    return provenance_roots[name].privacy_budget

def have_same_tag(pe1, pe2):
    return pe1.tag == pe2.tag

# should not be exposed
def clear_global_states():
    global provenance_roots
    global provenance_tag_counts

    provenance_roots = {}
    provenance_tag_counts = {}
