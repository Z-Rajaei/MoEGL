#main.py

import yaml
import os
from networkx.readwrite import json_graph
from extractElements import extractElements
from create_graph import CreateGraph
current_dir = os.path.dirname(__file__)
print("Current directory: ", current_dir)
config_file = os.path.join(current_dir, 'config.yaml')

with open(config_file) as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

try:
    format = config['inputmodels']['format']
    metamodelpath = config['inputmodels']['metamodelpath']
    modelpath = config['inputmodels']['modelspath']
    output = config['output']
    filters = config['adaptations']['metamodels']['packages']['ecore']['uri']['http://www.eclipse.org/emf/2002/Ecore']
    # print("classes", filters)

    print(format)
    print(metamodelpath)
    print(modelpath)
except KeyError:
  print("Section not found in YAML file")


csvpath =  modelpath.replace("models", "csvfiles")

if not os.path.exists(csvpath):

    graphinstance = extractElements(modelpath, metamodelpath)

graph = CreateGraph(csvpath,filters)
if output =="NetworkX":
    graphs = graph.model2HomoGraph()
    graphs_list = list(graphs.values())
    for graph in graphs_list:
        linkdata = json_graph.node_link_data(graph)
        print("grgaph is:",linkdata)
elif output =="PyG":
    print("maiiiiiiiiiiin heterograph is", graph.model2heterograph())





# input_file = config['input']
# output_file = config['output']

# extract(input_file, output_file)
