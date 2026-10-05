import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EcorePackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import com.google.gson.Gson;

import gg.core.GraphModel;
import gg.core.IMetaFilter;
import gg.core.MetaFilterLiterals;
import gg.core.Parser;

/**
 * Timing driver for the bespoke Ecore encoder of Lopez & Cuadrado (TCRMG-GNN).
 * It executes exactly the encoding steps of gg.main.GenerateRealGraphsEcore
 * (resource loading -> Parser.parse -> GraphModel -> Gson JSON) on every .ecore
 * file of a directory, without the corpus filtering / random sampling, because
 * the input directory already is the sampled set of 500 real models.
 *
 * Usage: java -cp <lib/*;classes> TimeLopezEcore <modelsDir> <outDir|-> 
 * Prints one line: "LOPEZ_ENCODE_MS <ms>"  (wall time of the encoding loop only)
 * and, if outDir != "-", writes <i>.json (same layout as realGraphs/Ecore/all).
 */
public class TimeLopezEcore {
    public static void main(String[] args) throws Exception {
        String modelsDir = args[0];
        String outDir = args.length > 1 ? args[1] : "-";

        EcorePackage.eINSTANCE.eClass();
        Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("*", new XMIResourceFactoryImpl());
        IMetaFilter mf = MetaFilterLiterals.getEcoreFilter();
        Parser parser = new Parser(mf);
        Gson gson = new Gson();

        File[] filesArr = new File(modelsDir).listFiles((d, n) -> n.endsWith(".ecore"));
        Arrays.sort(filesArr, (a, b) -> a.getName().compareTo(b.getName()));
        List<File> files = Arrays.asList(filesArr);

        long t0 = System.nanoTime();
        int i = 0;
        int nodes = 0, edges = 0;
        for (File f : files) {
            ResourceSet rs = new ResourceSetImpl();
            Resource r = rs.getResource(URI.createFileURI(f.getAbsolutePath()), true);
            GraphModel gm = parser.parse(r, f.getAbsolutePath());
            String json = gson.toJson(gm);
            nodes += gm.getNodes().size();
            edges += gm.getEdges().size();
            if (!outDir.equals("-")) {
                try (PrintWriter out = new PrintWriter(new File(outDir, f.getName().replace(".ecore", ".json")))) {
                    out.println(json);
                }
            }
            i++;
        }
        long t1 = System.nanoTime();
        System.out.println("LOPEZ_ENCODE_MS " + (t1 - t0) / 1_000_000.0);
        System.out.println("LOPEZ_MODELS " + i + " NODES " + nodes + " EDGES " + edges);
    }
}
