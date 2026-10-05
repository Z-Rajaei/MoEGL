from mmap import mmap
from numpy import ndarray
import pyecore
from pyecore.resources import ResourceSet, URI
from pyecore.ecore import EClass, EAttribute, EString, EObject
import pyecore.ecore as Ecore
import matplotlib.pyplot as plt
import networkx as nx
import os
import random
from pyecore.resources.xmi import XMIResource
from scipy.sparse import csr_matrix
import numpy
import itertools
# from netgan import decoder
# import utils
# import numpy as np
# import xlrd


def rollback_temporary_change(exp_ref):
    if len(exp_ref) > 0:
        for ref in exp_ref:
            ref.upperBound = ref.upperBound - 1


class ENCODE_M2G:
    id_iter = itertools.count()  # It Generates a new ID incrementally by each call
    map_class_iter = itertools.count()  # It Generates a new number incrementally by each call
    map_iter = itertools.count(1)  # It Generates a new number incrementally by each call (it starts from 1)
    name_iter = itertools.count(1)  # It Generates a new number incrementally by each call (it starts from 1)

    classes = {}  # all classes inside metamodel
    unregulated_inheritance = []  # all classes inside metamodel with unsolved parent
    objects = []  # An array including all objects/elements in xmi file
    matrix_of_graph: ndarray = []  # 2D-Matrix of corresponding graph
    enum_dict = {}  # A set of class's EEnum by enum name
    obj_attrs_dict = {}  # A set of class's Attributes by class name
    references_dictionary = {}  # A set of class references features by class name
    references_pair_dictionary = {}  # A set of pair class and their reference
    references_type_mapping = {}  # A dictionary that contains references mapped to numbers
    node_types = []  # 2D-Matrix containing node types(labels)
    true_containment_classes = []
    including_root = False  # Show that we consider root element as a node or not
    mm_root = []
    c_sparse_matrix = csr_matrix(numpy.zeros((10, 10)))

    def __init__(self, metamodel_name, model_name):
        self.load_model(metamodel_name, model_name)

    def load_model(self, metamodel_name, model_name):
        """
        :param model_name:
        :param metamodel_name:
        :return:
        """
        rset = ResourceSet()
        # resource = rset.get_resource(URI('../data/' + metamodel_name))
        resource = rset.get_resource(URI('data/' + metamodel_name))
        self.mm_root = resource.contents[0]
        exp_refs = self.check_for_bound_exception()
        rset.metamodel_registry[self.mm_root.nsURI] = self.mm_root
        self.extract_classes_references(self.mm_root)
        try:
            # resource = rset.get_resource(URI('../data/' + model_name))
            resource = rset.get_resource(URI('data/' + model_name))
            model_root = resource.contents[0]
            model_root._internal_id = next(self.id_iter)
            self.node_types.append([model_root._internal_id, self.classes[model_root.eClass.name]])
            self.extract_objects_form_model(model_root)
            output = self.create_matrix()
            # create_triple_file(output, model_name, self.node_types)
            adj_matrix = self.create_square_matrix(output)
            self.prepare_decoder_data()
            # self.show_details()
        except pyecore.valuecontainer.BadValueError:
            raise Exception("Sorry, Pyecore cannot pars the xmi file. please check the order of inside element.")
        rollback_temporary_change(exp_refs)

    def prepare_decoder_data(self):
        return self.mm_root, self.classes, self.obj_attrs_dict, self.references_type_mapping, \
               self.references_pair_dictionary, self.enum_dict, self.node_types

    def show_details(self):
        print("...................EClass mapping............")
        dictionary_items = self.classes.items()
        for item in dictionary_items:
            print("Class_type:", item[0], ": ", item[1])
        print("\n...................Nodes mapping...................")
        for h in self.node_types:
            print("Node_id:", h[0], "Node_type", h[1])
        print("\n..................Class attributes....................")
        for i in self.obj_attrs_dict:
            print("Class:", i, "Attributes:", self.obj_attrs_dict[i])
        print("\n...................Reference mapping...................")
        print("Mapping-> no relation: ", 0)
        for o in self.references_type_mapping:
            print("Mapping->", o, " : ", self.references_type_mapping[o])
        print("\n...................References dictionary...................")
        for i in self.references_dictionary:
            print("References_dictionary->", i, " : ", self.references_dictionary[i])
        print("\n...................True containment...................")
        for j in self.true_containment_classes:
            print("True_containment: ", j)

    def check_for_bound_exception(self):
        # if we have upperBound-lowerBound=1 then pyecore cannot get set of elements, so we will temporary
        # add 1 to upperBound
        exp_ref = []
        for e_class in self.mm_root.eClassifiers:
            if e_class.eClass.name is "EClass":
                for ref in e_class.eStructuralFeatures:
                    if ref.eClass.name == "EReference":  # Extracting inner relations
                        if hasattr(ref, "lowerBound") and hasattr(ref, "upperBound"):
                            if ref.lowerBound > 0 and ref.upperBound - ref.lowerBound == 1:
                                ref.upperBound = ref.upperBound + 1
                                exp_ref.append(ref)
        return exp_ref

    def extract_classes_references(self, metamodel_root):
        if self.mm_root.eClass.name != "EPackage":
            self.classes.update({self.mm_root.eClass.name: next(self.map_class_iter)})
        for e_class in metamodel_root.eClassifiers:
            if e_class.eClass.name is "EClass":
                if e_class.eStructuralFeatures.owner.name not in self.classes:
                    self.classes.update({e_class.eStructuralFeatures.owner.name: next(self.map_class_iter)})
                references, containment_classes, references_pair, obj_attrs = self.extract_single_class_references(
                    e_class)
                self.references_dictionary.update(references)
                self.references_pair_dictionary.update(references_pair)
                self.obj_attrs_dict.update(obj_attrs)
                append_items2list(containment_classes, self.true_containment_classes)
            elif e_class.eClass.name is "EEnum":
                # If there is any "EDataType", we can handel it here!
                literals = []
                for i in e_class.eLiterals:
                    literals.append(i)
                self.enum_dict.update({e_class.name: literals})

        #  Completing the reference dictionary by adding parent references that are not available at first
        for e_class in self.unregulated_inheritance:
            references, containment_classes, references_pair, obj_attrs = self.extract_single_class_references(e_class)
            self.references_dictionary.update(references)
            self.references_pair_dictionary.update(references_pair)
            self.obj_attrs_dict.update(obj_attrs)
            append_items2list(containment_classes, self.true_containment_classes)

    def extract_single_class_references(self, e_class):
        """"
        extract single class references and attributes
        :param e_class: The root class
        :return: extracted data
        """
        true_containment_classes = []
        inner_references = []
        pair_references = []
        obj_attrs_dict = []

        for ref in e_class.eStructuralFeatures:
            if ref.eClass.name == "EReference":  # Extracting inner relations
                pair_references.append([ref.eType.name, ref.name])
                inner_references.append(ref.name)
                if ref.name not in self.references_type_mapping:
                    self.references_type_mapping.update({ref.name: next(self.map_iter)})
                if ref.containment:
                    true_containment_classes.append(ref.name)
            elif ref.eClass.name == "EAttribute":
                obj_attrs_dict.append([ref.name, ref.eType.name])

        # Creating references dictionary for the class
        references_dictionary = {e_class.name: inner_references}
        references_pair_dictionary = {e_class.name: pair_references}
        obj_attrs_dictionary = {e_class.name: obj_attrs_dict}
        self.add_inheritance_references(e_class, references_dictionary)

        return references_dictionary, true_containment_classes, references_pair_dictionary, obj_attrs_dictionary

    def add_inheritance_references(self, e_class, references_dictionary):
        if len(e_class.eSuperTypes.items) > 0:  # It means the e_class has a parent
            if hasattr(e_class.eSuperTypes.items[0], "items"):  # It means the parent has another parent
                self.add_inheritance_references(e_class.eSuperTypes.items[0], references_dictionary)
            else:
                if e_class.eSuperTypes.items[0].name in self.references_dictionary:
                    # if parent is already in dictionary
                    references_dictionary.update(
                        {e_class.name: self.references_dictionary[e_class.eSuperTypes.items[0].name]})
                else:
                    # if parent doesn't exist in dictionary, add parent to dictionary
                    self.unregulated_inheritance.append(e_class)

    def extract_objects_form_model(self, root_object):
        """
        :param root_object: The root class
        """
        for class_name in self.true_containment_classes:
            if hasattr(root_object, class_name):
                all_instances_by_same_class_name = getattr(root_object, class_name)
            else:
                continue
            if hasattr(all_instances_by_same_class_name, "items"):  # if we have several items in same type
                for obj in all_instances_by_same_class_name:
                    if obj not in self.objects:
                        obj._internal_id = next(self.id_iter)  # Getting an ID for assigning to each element
                        self.node_types.append([obj._internal_id, self.classes[obj.eClass.name]])
                        self.objects.append(obj)
                        for inner_class_name in self.true_containment_classes:
                            if hasattr(obj, inner_class_name):
                                self.extract_objects_form_model(obj)
            elif all_instances_by_same_class_name is not None:  # if we have just one item
                all_instances_by_same_class_name._internal_id = next(
                    self.id_iter)  # generating an ID for assign to the element
                self.node_types.append([all_instances_by_same_class_name._internal_id,
                                        self.classes[all_instances_by_same_class_name._containment_feature.eType.name]])
                self.objects.append(all_instances_by_same_class_name)
                for inner_class_name in self.true_containment_classes:
                    if hasattr(all_instances_by_same_class_name, inner_class_name):
                        self.extract_objects_form_model(all_instances_by_same_class_name)

    def create_matrix(self):
        # Create an empty 2D[len(input),len(input)] array as a matrix
        self.matrix_of_graph = numpy.zeros(
            (len(self.objects) + 1, len(self.objects) + 1))
        for obj in self.objects:
            self.seek_in_depth(obj, self.references_dictionary)
        for i in range(0, len(self.matrix_of_graph)):
            for j in range(0, len(self.matrix_of_graph)):
                if self.matrix_of_graph[i][j] > 0:
                    self.matrix_of_graph[j][i] = self.matrix_of_graph[i][j]
        return self.matrix_of_graph

    def seek_in_depth(self, obj, references_dictionary):
        """
        Checking inside relations and adding corresponding edges
        :param obj: The object that we want to extract all elements inside it
        :param references_dictionary: set of class references features by class name
        :return:
        """
        inner_references_name = references_dictionary[obj.eClass.name]
        if len(inner_references_name) == 0:  # if we just have relations between root and other elements
            self.check_and_add_relations_with_root(obj)
        for inner_ref_name in inner_references_name:
            if hasattr(obj, inner_ref_name):
                inner_element = getattr(obj, inner_ref_name)
                if inner_element is not None:  # check if references feature is not empty, try to find relation (edges)
                    if hasattr(inner_element, '_internal_id'):  # If we have a single inside element
                        if inner_element._internal_id is not None:
                            self.matrix_of_graph[obj._internal_id][inner_element._internal_id] = \
                                self.references_type_mapping[inner_ref_name]
                    else:  # If we have a set of inside elements
                        if self.matrix_of_graph[obj._container._internal_id][
                            obj._internal_id] == 0 and obj._container._internal_id == 0:
                            self.check_and_add_relations_with_root(obj)
                        set_elements = inner_element.items
                        for i in set_elements:
                            if i._internal_id is not None:
                                if not (self.matrix_of_graph[obj._internal_id][i._internal_id] > 0):
                                    self.matrix_of_graph[obj._internal_id][i._internal_id] = \
                                        self.references_type_mapping[
                                            inner_ref_name]
                                if i._container._internal_id == 0:
                                    self.check_and_add_relations_with_root(i)

    def check_and_add_relations_with_root(self, obj):
        """
        check and add relations with root
        :param obj: an inner object existing in model
        """
        # Add an relation type for the root's relations
        self.matrix_of_graph[obj._container._internal_id][obj._internal_id] = self.references_type_mapping[
            obj._containment_feature.name]

    def create_square_matrix(self, matrix):
        """
        convert input matrix to a square matrix
        :param matrix: adjacency matrix
        :return:
        """
        for idx, row in enumerate(matrix):
            for idy, val in enumerate(row):
                if val > 0:
                    matrix[idx, idy] = 1
                    matrix[idy, idx] = 1
        self.c_sparse_matrix = csr_matrix(matrix)
        print("Matrix for NetGAN: ", self.c_sparse_matrix.count_nonzero(), "\n", matrix)
        return matrix


def create_triple_file(matrix, ds_name, labels):
    """
    prepare data for he_gan
    :param matrix: adjacency matrix
    :param ds_name: data set name
    :param labels: list of node labels (object types)
    """
    if ds_name.endswith('.xmi'):
        ds_name = ds_name[:-4]
    first_line = True
    with open('../output_DS/' + ds_name + '_triple.dat', 'w') as f:
        for idx, row in enumerate(matrix):
            for idy, val in enumerate(row):
                if val > 0:
                    line = "\n" + str(idx) + " " + str(idy) + " " + str(int(val))
                    if first_line:
                        line = str(idx) + " " + str(idy) + " " + str(int(val))
                    f.write(line)
                    first_line = False
    first_line = True
    with open('../output_DS/' + ds_name + '_label.dat', 'w') as file:
        for i in labels:
            line = "\n" + str(i[0]) + "  " + str(i[1])
            if first_line:
                line = str(i[0]) + "  " + str(i[1])
            file.write(line)
            first_line = False


def look_inside(obj):
    """
    :param obj: The object that we want to look elements that are inside it
    """
    print(".............. Checking inside object .................")
    temp = vars(obj)
    for item in temp:
        print(item, ':', temp[item])
    print(".......................................................")


def append_items2list(input_list, cumulative_list):
    for item in input_list:
        cumulative_list.append(item)
    return cumulative_list
