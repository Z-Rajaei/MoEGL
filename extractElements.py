import csv
import glob
import os
from collections.abc import Iterable
# from org.eclipse.emf import ecore
from pyecore.ecore import EAttribute
from pyecore.ecore import EStructuralFeature
from pyecore.ecore import EOrderedSet
from pyecore.ecore import EClass


# from ecore import EAttribute


# import networkx as nx
import yaml
from pyecore.ecore import EEnum
from pyecore.resources import ResourceSet, URI


class extractElements:
    case = "xmi"
    ecoreNumber = 0
    featureTypes = []
    graphNo = 0
    nodeLabel = 1
    Graphs = []
    attributeNames = {}
    classes = dict()
    directories = 1
    filenumber = 0
    yamlexistance = 0
    mapvalue = {}
    yamlcontents = {}
    referenceNames = {}
    IDs = {}
    graphAttributes = {}
    visitednodes = []
    jsondata = {}
    visited = []
    queue = []
    pathfile = None
    yamlFeatures = {}
    yamlclasses = {'include': [], 'exclude': []}
    modelData = {}
    csvData = {}
    refFeatures = []
    csvEdgesData = {}
    folder = ""
    ids = {}

    graphNumber = 0

    def __init__(self, models_path: str, metamodel_path:str):
        self.models_path = models_path
        self.metamodel_path = metamodel_path

        self.load_config()
        self.load_metamodel(metamodel_path)
        self.process_models(models_path)

        #check the format of input models in Yaml
    def load_config(self) -> None:

        self.current_dir = os.path.dirname(__file__)
        config_file = os.path.join(self.current_dir, 'config.yaml')
        with open(config_file) as f:
            self.yamlcontents = yaml.load(f, Loader=yaml.FullLoader)
        self.case = self.yamlcontents['inputmodels']['format']

    def load_metamodel(self, metamodel_path) -> None:
        rset = ResourceSet()
        resource = rset.get_resource(URI(metamodel_path))
        self.meta_root = resource.contents[0]

    def process_models(self, models_path) -> None:
        if self.case == "ecore":
            self.load_ecore_models(models_path)
        elif self.case == "xmi":
            self.load_xmi_models(models_path)
        self.create_edges()
        self.create_csv()

    def load_ecore_models(self, models_path):

        os.chdir(models_path)

        for file in glob.glob("*.ecore"):
            print("file is", file)
            ecore_file = os.path.join(models_path, file)

            self.rset = ResourceSet()
            resource = self.rset.get_resource(URI(ecore_file))
            model_root = resource.contents[0]

            self.ecoreNumber += 1
            self.create_nodes(model_root, self.meta_root)

    def load_xmi_models(self, models_path):

        os.chdir(models_path)

        self.rset.metamodel_registry[self.meta_root.nsURI] = self.meta_root

        for file in glob.glob("*.xmi"):
            xmi_file = os.path.join(models_path, file)

            resource = self.rset.get_resource(URI(xmi_file))
            model_root = resource.contents[0]

            self.filenumber += 1
            self.create_nodes(model_root, self.meta_root)



    def checkYaml(self, mmroot, model, elementType, case, element=None, ):
        print("checkyaml is", "mmroot=", mmroot, "model is=", model, "elementType=", elementType, "case=", case,
              "element=", element)
        metamodel_name = mmroot.name
        metamodel_nsURI = mmroot.nsURI
        modelName = model.eClass.name
        print("element is=", element)
        if element != None:
            elementName = element.name
        # print("model in checkyaml:", modelName)
        if self.yamlexistance == 0:
            print("yaml 0")
            return True
        elif case == "include" and elementType == "class":
            print("checking existance")

            print("meta name=", metamodel_name)
            print("ns is", metamodel_nsURI)
            print("modelName=", modelName)
            if ((self.deep_get(self.yamlcontents,
                               ['adaptations', 'metamodels', 'packages', metamodel_name,
                                'uri', mmroot.nsURI, 'classes',
                                'exclude']) != None and modelName in self.deep_get(
                self.yamlcontents,
                ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri', mmroot.nsURI, 'classes',
                 'exclude'])) or (self.deep_get(self.yamlcontents,
                                                ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri',
                                                 mmroot.nsURI, 'classes',
                                                 'include']) != None and modelName not in self.deep_get(
                self.yamlcontents,
                ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri', mmroot.nsURI, 'classes',
                 'include']))):
                print("model True")
                return False
            else:
                return True
        elif elementType == "feature" and case == "existance":
            print("here feature existance")
            if self.deep_get(self.yamlcontents,
                             ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri', mmroot.nsURI,
                              'classes', modelName, 'features', elementName, 'isIncluded']) != None:
                if self.deep_get(self.yamlcontents,
                                 ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri', mmroot.nsURI,
                                  'classes', modelName, 'features', elementName, 'isIncluded']) == "False":
                    print("return False1")
                    return False





            elif (self.deep_get(self.yamlcontents,
                                ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri', mmroot.nsURI,
                                 'classes', modelName, "include"]) != None and elementName not in self.deep_get(
                self.yamlcontents, ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri', mmroot.nsURI,
                                    'classes', modelName, "include"])):
                print("return False2")
                return False
            elif self.deep_get(self.yamlcontents,
                               ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri', mmroot.nsURI,
                                'classes', "excludeAllAttributes"]) == "True":
                print("return False3")
                return False
            else:
                print("return True")
                return True
        elif case == "renaming":
            print("yamlcontent::", self.yamlcontents)
            print("renaming")
            print("metaname:", metamodel_name)
            print("model", modelName)
            print("element:", elementName)
            print("rename value==", self.deep_get(self.yamlcontents,
                                                  ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri',
                                                   mmroot.nsURI,
                                                   'classes', modelName, 'features', elementName, 'renaming']))
            if self.deep_get(self.yamlcontents,
                             ['adaptations', 'metamodels', 'packages', metamodel_name, 'uri', mmroot.nsURI,
                              'classes', modelName, 'features', elementName, 'renaming']) != None:
                print("rename exists")
                return self.yamlcontents['adaptations']['metamodels']['packages'][metamodel_name]['uri'][mmroot.nsURI][
                    'classes'][modelName]['features'][elementName]['renaming']
            else:

                return True

    def deep_get(self, d, keys):
        if not keys or d is None:
            return d
        return self.deep_get(d.get(keys[0]), keys[1:])

    model_data = {}
    attribute_types = []

    def create_nodes(self, model, mmroot):
        print("creating nodes")
        self.graphNumber = self.graphNumber + 1
        print("graphNumber issss:", self.graphNumber)
        # Data structures
        self.model_data = {}
        self.attribute_types = []

        # Create node data for root model

        node_data = self.create_node_data(model, mmroot, self.graphNumber)


        # Add root node data to model data
        self.add_to_model_data(model, node_data)

        # Process child elements
        for element in self.get_child_elements(model):
            print("child element")
            # Create node data for child

            child_node_data = self.create_node_data(element, mmroot, self.graphNumber)

            # Add child node data to model data
            self.add_to_model_data(element, child_node_data)

        # Return data structures
        return self.model_data, self.attribute_types

    def create_node_data(self, element, mmroot, graphNo):
        print("creating node data")

        graphdata = {}
        graphcsvdata = {}

        # metamodel_name = mmroot.name
        # modelName = element.eClass.name

        node_number = 0
        if element.eClass.name in self.ids:
            node_number = self.ids[element.eClass.name] + 1

        self.ids[element.eClass.name] = node_number

        for feature in element._isset:


            feature_name = feature.name
            feature_value = element.eGet(feature)

            graphdata[feature_name] = feature_value
            # print("graphNo is now:", graphNo)
            graphdata["graphNumber"] = graphNo
            graphcsvdata["graphNumber"] = graphNo

            if not feature.is_reference:
                graphcsvdata[feature_name] = feature_value

            if feature.is_attribute:

                types = {}
                types["node"] = element.eClass.name
                types["name"] = feature.name
                print("ids is:", self.ids)
                print("element is", element)
                types["nodeNumber"] = self.ids[element.eClass.name]
                # node_number = self.ids.get(class_name, 0) + 1

                if isinstance(feature.eType, EEnum):
                    types["type"] = "EString"
                else:
                    types["type"] = feature.eType.name

                self.featureTypes.append(types)
                types = {}

        self.modelData.setdefault(element.eClass.name, []).append(graphdata)
        self.modelData[element.eClass.name][-1]["element"] = element

        self.csvData.setdefault(element.eClass.name, []).append(graphcsvdata)


        return graphdata



    def get_node_number(self, element):
        print("csvdata", self.csvData)
        print(" modeldata", self.modelData)
        class_name = element.eClass.name

        node_number = self.ids.get(class_name, 0) + 1

        self.ids[class_name] = node_number

        return node_number

    def add_to_model_data(self, element, data):
        print("add to model data")


        self.model_data.setdefault(element.eClass.name, []).append(data)
        data["element"] = element

    def get_child_elements(self, element):

        return element.eAllContents()

    def get_eclass(self, element, mmroot):

        return mmroot.getEClassifier(element.eClass.name)

    def get_features(self, eclass):

        return list(eclass.eStructuralFeatures)



    def create_edges(self):
        print("Creating edges...")
        for keys in self.csvData:
            for i in range(0, len(self.csvData[keys])):
                        id = str(keys) + "Id"
                        self.csvData[keys][i][id] = i
                        self.modelData[keys][i][id] = i

        reference_properties = {"eClassifiers", "ePackage", "eStructuralFeatures", "eType",
                        "eContainingClass", "eOpposite", "eSuperTypes", "eModelElement"}

        for node_type, nodes in self.modelData.items():
            # print("nodes is", nodes)
            for node in nodes:

                for property, references in node.items():
                    if property in reference_properties:

                        # print("references", references)
                        if isinstance(references, Iterable):
                            # print("yes iterable")
                            refs = list(references)
                        else:
                            refs = [references]

                        source_id = node[f"{node_type}Id"]
                        graph_number = node["graphNumber"]
                        # print("references", refs)
                        for reference in refs:

                            edge_type = reference.eClass.name

                            if self.csvEdgesData.get(f"{node_type}_{edge_type}") is None:
                                self.csvEdgesData[f"{node_type}_{edge_type}"] = []
                            # print("edge type", edge_type)
                            # print("model data:", self.modelData)
                            if edge_type in self.modelData:

                                for dest_node in self.modelData[edge_type]:

                                    if ("element", reference) in dest_node.items() and (
                                    dest_node["graphNumber"], graph_number):
                                        dest_id = dest_node[f"{edge_type}Id"]

                                        edge_data = {}
                                        edge_data[f"sr{node_type}Id"] = source_id
                                        edge_data[f"ds{edge_type}Id"] = dest_id
                                        edge_data["name"] = property

                                        self.csvEdgesData[f"{node_type}_{edge_type}"].append(edge_data)

    import csv
    import os

    def create_csv(self):



        print("Creating CSV files...")

        # Encapsulate node logic
        self._write_nodes_csv()

        # Encapsulate edges logic
        self._write_edges_csv()

        # Encapsulate types logic
        self._write_types_csv()

        print("CSV files created!")



    def _write_nodes_csv(self):
        for types, records in self.csvData.items():
            self._write_to_csv(records, types, "nodes")

    def _write_edges_csv(self):
        for types, records in self.csvEdgesData.items():
            if records:
                self._write_to_csv(records, types, "edges")

    def _write_types_csv(self):
        print("feature types:", self.featureTypes)
        self._write_to_csv(self.featureTypes, "types", folder="types")

    def _write_to_csv(self, records, types, folder):
        if types == "types":
            filename = f"{types}"
        else:
            filename = f"{types}_{folder}"
        dir_path = f"..\\csvfiles\\{folder}"

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        fields = records[0].keys()

        with open(os.path.join(dir_path, f'{filename}.csv'), 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()

            writer.writerows(records)

