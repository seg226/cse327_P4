from copy import copy
import os
import pickle
import sys

from basictypes import Atom, Predicate, Variable, Constant
import numpy as np

from knowledgebase import Rule

import torch

MIN_VARS = 10


class Vocabulary:
    """ Defines a first order logic set of symbols. This class is not yet used,
    but will eventually be integrated into the process of generating random
    knowledge bases, generating one hot vectors, generating triplet loss training
    files, and translating from one set of symbols to another.
    """
    #If None is passed (or they are not provided, due to the default value of None), these parameters are initialized as empty lists
    def __init__(self, predicates: list[Predicate] | None = None, constants: list[Constant] | None = None, variables: list[Variable] | None = None):
        # if preds_by_arity is None:
        #    preds_by_arity = []
        if predicates is None:
            predicates = []
        if constants is None:
            constants = []
        if variables is None:
            variables = []

        self.predicates = predicates
        self.constants = constants
        self.variables = variables
        #It categorizes each predicate by its arity (the number of arguments the predicate takes).
        # This is useful for quickly accessing all predicates of a certain arity. ?
        self.predicatesByArity: dict[int, list[Predicate]] = {}
        for pred in self.predicates:
            if pred.arity not in self.predicatesByArity:
                self.predicatesByArity[pred.arity] = []
            self.predicatesByArity[pred.arity].append(pred)
        # self.predicatesByArity is sorted by key (arity), which makes retrieval by arity efficient and organized.
        self.predicatesByArity = dict(sorted(self.predicatesByArity.items()))
        #Calculates the maximum arity among all predicates by taking the max of the keys in self.predicatesByArity dictionary.
        self.maxArity = max(self.predicatesByArity.keys()) if len(
            self.predicatesByArity) > 0 else 0

    #num_pred: The number of random predicates to generate.
    #arity_dist: A list representing the distribution of arities for the generated predicates. Each element's index represents arity-1, and its value represents the probability of that arity.
    #num_const: The number of random constants to generate.
    #num_var: The number of random variables to generate.
    #
    def random_init(self, num_pred=10, arity_dist=None, num_const=100, num_var=10):
        """ Initialize with a randomly generated set of symbols. This will overwrite any symbols
        already present. """
        # This method provides a way to populate the object
        # with a specified number of random predicates, constants, and variables,
        # with an option to specify the distribution of predicate arities.

        # arity_dist gives the probability of arity being index+1
        #corresponds to a preference for predicates with lower arities.
        if arity_dist == None:
            arity_dist = [0.3, 0.3, 0.2, 0.1, 0.1]
        # generate random predicates
        for i in range(num_pred):
            #Randomly selects an arity based on arity_dist.
            #Creates a Predicate object with the selected arity and a name based on its index ("p" + str(i)).
            arity = np.random.choice(range(len(arity_dist)), p=arity_dist) + 1
            self.predicates.append(Predicate(arity, "p" + str(i)))

        # Constants and Variables Generation: Generates num_const constants and num_var variables
        # with predefined naming conventions ("a" + str(i) for constants and "X" + str(i) for variables)
        constants = [Constant("a" + str(i)) for i in range(num_const)]
        variables = [Variable("X" + str(i)) for i in range(num_var)]

        self.constants = sorted(constants, key=lambda x: x.name)
        self.predicates = sorted(self.predicates, key=lambda x: x.name)
        self.variables = sorted(variables, key=lambda x: x.name)

        # add class data member for predicates by arity
        for pred in self.predicates:
            if pred.arity not in self.predicatesByArity:
                self.predicatesByArity[pred.arity] = []
            self.predicatesByArity[pred.arity].append(pred)
        self.predicatesByArity = dict(sorted(self.predicatesByArity.items()))
        self.maxArity = max(self.predicatesByArity.keys())

    def init_from_kb(self, kb):
        """ Initialize symbols from a knowledge base. This will overwrite any symbols already present. """
        # This method updates the objects lists of predicates, constants, and variables by extracting them from the knowledge base's rules

        # Each rule is assumed to have a head and a body, where the head contains a single predicate(the conclusion)
        # and the body contains a list of predicates(the premises).
        for rule in kb.rules:
            if rule.head.predicate not in self.predicates:
                self.predicates.append(rule.head.predicate)
            for arg in rule.head.arguments:
                if isinstance(arg, Constant) and arg not in self.constants:
                    self.constants.append(arg)
                elif isinstance(arg, Variable) and arg not in self.variables:
                    self.variables.append(arg)
            for atom in rule.body:
                if atom.predicate not in self.predicates:
                    self.predicates.append(atom.predicate)
                for arg in atom.arguments:
                    if isinstance(arg, Constant) and arg not in self.constants:
                        self.constants.append(arg)
                    elif isinstance(arg, Variable) and arg not in self.variables:
                        self.variables.append(arg)

        self.constants = sorted(self.constants, key=lambda x: x.name)
        self.predicates = sorted(self.predicates, key=lambda x: x.name)
        self.variables = sorted(self.variables, key=lambda x: x.name)
        #A mechanism to ensure a minimum number of variables(MIN_VARS) in self.variables by adding new, uniquely named variables if needed.
        i = 0
        while len(self.variables) < MIN_VARS:
            new_variable = Variable("_EX" + str(i))
            if new_variable not in self.variables:
                self.variables.append(new_variable)
            i += 1

        # add class data member for predicates by arity
        #categorizing all predicates by their arity
        for pred in self.predicates:
            if pred.arity not in self.predicatesByArity:
                self.predicatesByArity[pred.arity] = []
            self.predicatesByArity[pred.arity].append(pred)
        self.predicatesByArity = dict(sorted(self.predicatesByArity.items()))
        self.maxArity = max(self.predicatesByArity.keys())

    def get_one_hot_size(self) -> int:
        """

        :return: The length of a one hot encoding of the vocabulary
        """
        total_terms = len(self.constants) + len(self.variables)
        # length depends on the number of predicates and the maximum arity of any predicate multiplied by total_terms.
        return len(self.predicates) + total_terms * self.maxArity


    def get_by_arity(self, arity):
        """ Returns a list of predicates with the given arity. """

        return self.predicatesByArity[arity]

    def oneHotEncoding(self, atom: Atom, device="cpu") -> torch.Tensor:
        """
        Given an atom, returns a one-hot encoding that is consistent with the vocabulary.
        The encoding is actually the concatenation of several one-hot vectors in the
        order [predicate, term1, term2, etc.]. The one-hot vectors for terms are ordered
        by variables and then constants.
        :param atom:
        :param device:
        :return: A PyTorch Tensor on the desired device
        """

        total_terms = len(self.constants) + len(self.variables)
        #a zero vector of a length that depends on the number of predicates and the maximum arity of any predicate multiplied by total_terms.
        encoding = torch.zeros(self.get_one_hot_size(), device=device)
        # encoding = [0] * self.get_one_hot_size()

        #Encoding the Predicate
        #Yifan check: whether is all of indexes
        encoding[self.predicates.index(atom.predicate)] = 1

        #Encoding the Arguments
        start = len(self.predicates)
        for arg in atom.arguments:
            if isinstance(arg, Constant):
                if arg not in self.constants:
                    print("Error: Constant not found in vocabulary: " + arg.name)
                    exit(1)
                move = len(self.variables) + self.constants.index(arg)
            else:
                if arg not in self.variables:
                    print("Error: Variables not found in vocabulary: " + arg.name)
                    exit(1)
                move = self.variables.index(arg)
            #if Constant: offset =  the number of predicates and variables (start + len(self.variables) + index).
            ##if variables: offset =  the number of predicates  (start +  index).
            encoding[start+move] = 1
            start += total_terms
        return encoding

    # output vocab file

    # if no filename is provided when the method is called, it will default to saving the file as "vocab.pkl"
    def save_vocab_to_file(self, filename: str = "vocab"):
        with open(filename + '.pkl', 'wb') as handle:
            #the highest available protocol should be used for pickling. This can make the pickling process more efficient
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print("Vocabulary saved to " + filename + '.pkl')

    # init from vocab file
    def init_from_vocab(self, filename: str = "vocab"):
        if not os.path.isfile(filename + '.pkl'):
            print("No file found for vocab load: " + filename + ".pkl")
            sys.exit(1)
        with open(filename + '.pkl', 'rb') as handle:
            loaded: Vocabulary = pickle.load(handle)
            self.predicates = loaded.predicates
            self.constants = loaded.constants
            self.variables = loaded.variables
            self.predicatesByArity = loaded.predicatesByArity
            self.maxArity = loaded.maxArity
        # print("Vocabulary loaded from " + filename + '.pkl')

    def sanitize_rule(self, item: Rule):
        """ Returns a sanitized version of the given rule, replacing
        any symbols not in the vocabulary with new symbols. """
        #This method is designed to process a given Rule object by ensuring all its variables are within the vocabulary defined by the current object
        #If the rule contains variables not present in the object's vocabulary, the method replaces them with variables from the vocabulary that are not currently used in the rule
        # new_item = deepcopy(item)

        variables = {}
        notInVocab = []
        new_vars = []
        inRule = set()

        inRule.update([
            var for var in item.head.arguments if isinstance(var, Variable)])
        for atom in item.body:
            inRule.update([var for var in atom.arguments if isinstance(var, Variable)])
        for var in inRule:
            if var not in self.variables:
                notInVocab.append(var)

        # JDH 4/28/24: Note, rule.vars used to just be the names of variables!
        # I changed it there, but it was still giving bad results. I added this
        # notInRule code
        # TODO: for efficiency, it might make sense to move this to rule, as long as we keep it updated here
        notInRule = [
            var for var in self.variables if var not in inRule]
        if len(notInVocab) > len(notInRule):
            print("Error: Not enough variables in vocabulary")
            exit(1)
        for i in range(len(notInVocab)):
            variables[notInVocab[i]] = notInRule[i]

        # substitute the in-vocab variables for the not-in vocab ones
        new_item =  Rule(item.head.dosubst(variables), [x.dosubst(variables) for x in item.body])


        # replaced this with simpler version above
        # Note: this appears to leave new_item's var field in an inconsistent state. It is never updated with the renamed variables
        # for i in range(len(new_item.head.arguments)):
        #     arg_i = new_item.head.arguments[i]
        #     if isinstance(arg_i, Variable):
        #         if arg_i in notInVocab:
        #             new_item.head.arguments[i] = variables[arg_i]
        #             new_vars.append(variables[arg_i])
        #         elif isinstance(new_item.head.arguments[i], Variable):
        #             new_vars.append(new_item.head.arguments[i])
        # for i in range(len(new_item.body)):
        #     for j in range(len(new_item.body[i].arguments)):
        #         if isinstance(new_item.body[i].arguments[j], Variable) and new_item.body[i].arguments[j] in notInVocab:
        #             new_item.body[i].arguments[j] = variables[new_item.body[i].arguments[j]]
        #             new_vars.append(
        #                 variables[new_item.body[i].arguments[j].name])
        #         elif isinstance(new_item.body[i].arguments[j], Variable):
        #             new_vars.append(new_item.body[i].arguments[j])

        return new_item

    def sanitize_atom(self, item: Atom):
        """
        Returns a sanitized version of the given atom, replacing
        any symbols not in the vocabulary with new symbols. The
        original atom will not be changed.

        :param item:
        :return:
        """

        new_item = copy(item)

        variables = {}
        inAtom = [
            var for var in new_item.arguments if isinstance(var, Variable)]
        notInVocab = [var for var in inAtom if var not in self.variables]
        notInAtom = [var for var in self.variables if var not in inAtom]
        if len(notInVocab) > len(notInAtom):
            print("Error: Not enough variables in vocabulary")
            exit(1)
        for i in range(len(notInVocab)):
            variables[notInVocab[i]] = notInAtom[i]
        for i in range(len(new_item.arguments)):
            if isinstance(new_item.arguments[i], Variable) and new_item.arguments[i] in notInVocab:
                new_item.arguments[i] = variables[new_item.arguments[i]]

        return new_item


    def print_summary(self):
        print("Num predicates: " + str(len(self.predicates)))
        print("Max arity: " + str(self.maxArity))
        print("Num constants: " + str(len(self.constants)))
        print("Num variables: " + str(len(self.variables)))


if __name__ == "__main__":
    my_vocab = Vocabulary()
    my_vocab.init_from_vocab("vocab")
    my_vocab.print_summary()
    print()
    pred_list = [p.get_pred_arity_str() for p in my_vocab.predicates]
    print(pred_list)
    if len(my_vocab.constants) < 20:
        print(my_vocab.constants)
    else:
        print(str(my_vocab.constants[0:20]) + ", ...")
    print(my_vocab.variables)
