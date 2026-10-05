import java.io.File;
import java.io.FileInputStream;
import java.io.PrintWriter;
import java.util.Arrays;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EcorePackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceImpl;

import com.google.gson.Gson;

import gg.core.GraphModel;
import gg.core.IMetaFilter;
import gg.core.MetaFilterLiterals;
import gg.core.MetaFilterNames;
import gg.core.Parser;

/**
 * Timing driver for the bespoke encoders of Lopez & Cuadrado (TCRMG-GNN),
 * for all three datasets of their evaluation.
 *
 * It reproduces, inside the timed loop, exactly what their mains do:
 *   Ecore   : gg.main.GenerateRealGraphsEcore   (MetaFilterLiterals.getEcoreFilter)
 *   RDS     : gg.main.GenerateRealGraphsRDS      (RdsFull loader + getRDSFilter)
 *   Yakindu : gg.main.GenerateRealGraphsYakindu  (YakinduFullLoader + getYakinduFilter)
 * minus corpus filtering / sampling (the input dir already is the released set),
 * and minus writing files unless outDir is given.
 *
 * Their loaders register the dataset metamodel(s) in the GLOBAL package registry
 * once (static cache) and then parse every file with a fresh XMIResourceImpl --
 * the registration cost lands inside the timed loop on the first file, so we do
 * the same. The only fix vs their code: metamodel paths are taken from an arg
 * instead of the hard-coded /home/antolin/... absolute paths in
 * YakinduFullLoader (which make the code fail on any other machine).
 *
 * Usage:
 *   java -cp <lib/*;classes> TimeLopezAll <ecore|rds|yakindu> <modelsDir> <mmDir> <outDir|->
 *     <mmDir>: directory containing the dataset metamodel(s) (ignored for ecore)
 * Prints "LOPEZ_ENCODE_MS <ms>" and "LOPEZ_MODELS <n> NODES <n> EDGES <n>".
 */
public class TimeLopezAll {

    private static boolean metamodelsRegistered = false;

    /** Register all .ecore files of mmDir in the global package registry,
     *  mirroring gg.loaders.RdsFull / YakinduFullLoader.initYakindu(). */
    private static void registerMetamodels(File mmDir) {
        if (metamodelsRegistered) return;
        ResourceSet rs = new ResourceSetImpl();
        File[] ecores = mmDir.listFiles((d, n) -> n.endsWith(".ecore"));
        Arrays.sort(ecores);
        for (File mm : ecores) {
            try {
                Resource r = rs.getResource(URI.createFileURI(mm.getAbsolutePath()), true);
                r.getAllContents().forEachRemaining(o -> {
                    if (o instanceof EPackage) {
                        EPackage.Registry.INSTANCE.put(((EPackage) o).getNsURI(), (EPackage) o);
                    }
                });
            } catch (Exception e) {
                // a metamodel that fails to load is skipped, like their loaders do
            }
        }
        metamodelsRegistered = true;
    }

    public static void main(String[] args) throws Exception {
        String dataset = args[0];                    // ecore | rds | yakindu
        File modelsDir = new File(args[1]);
        File mmDir = args.length > 2 && !args[2].equals("-") ? new File(args[2]) : null;
        String outDir = args.length > 3 ? args[3] : "-";

        final String ext = dataset.equals("yakindu") ? ".sct" : (dataset.equals("rds") ? ".xmi" : ".ecore");

        EcorePackage.eINSTANCE.eClass();
        Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("*", new XMIResourceFactoryImpl());

        IMetaFilter mf;
        switch (dataset) {
            case "ecore":   mf = MetaFilterLiterals.getEcoreFilter(); break;
            case "rds":     mf = MetaFilterNames.getRDSFilter();      break;
            case "yakindu": mf = MetaFilterNames.getYakinduFilter();  break;
            default: throw new IllegalArgumentException("dataset must be ecore|rds|yakindu");
        }
        Parser parser = new Parser(mf);
        Gson gson = new Gson();

        File[] filesArr = modelsDir.listFiles((d, n) -> n.endsWith(ext));
        Arrays.sort(filesArr, (a, b) -> a.getName().compareTo(b.getName()));

        long t0 = System.nanoTime();
        int i = 0, nodes = 0, edges = 0;
        for (File f : filesArr) {
            Resource r;
            if (dataset.equals("ecore")) {
                ResourceSet rs = new ResourceSetImpl();
                r = rs.getResource(URI.createFileURI(f.getAbsolutePath()), true);
            } else {
                // RdsFull / YakinduFullLoader: metamodels once (lazy, inside the
                // timed loop like in their code), then a bare XMIResourceImpl.
                if (mmDir != null) registerMetamodels(mmDir);
                r = new XMIResourceImpl();
                try {
                    r.load(new FileInputStream(f), null);
                } catch (Exception e) {
                    // their loaders return the (possibly empty) resource on error
                }
            }
            GraphModel gm = parser.parse(r, f.getAbsolutePath());
            if (gm.getNodes().size() < 4) continue;   // same >=4 filter as their mains
            String json = gson.toJson(gm);
            nodes += gm.getNodes().size();
            edges += gm.getEdges().size();
            if (!outDir.equals("-")) {
                try (PrintWriter out = new PrintWriter(new File(outDir,
                        f.getName().replace(ext, ".json")))) {
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
