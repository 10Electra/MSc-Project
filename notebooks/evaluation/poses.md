
## 1 No Noise

### Drill
cam_offset=obj_centre
look_at = obj_centre
width: int = 360
height: int = 240
fov: float = 70.0
k: float = 10
max_normal_angle_deg = None
N = 8
radius = 0.28
include_depth_images = True
"capturespherical"
"fibonnacci"
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ -0.047563224652026517, 0.046038004086583227, 0.90710885961665622 ],
			"boundingbox_min" : [ -0.18797298036669985, -0.10495374720145056, 0.71914974875969284 ],
			"field_of_view" : 60.0,
			"front" : [ -0.74561220115033233, -0.65201208009554645, 0.13763245585702225 ],
			"lookat" : [ -0.11946817567763685, -0.027750299939340785, 0.81205313332433826 ],
			"up" : [ 0.10800519023380389, 0.085565323137221094, 0.99046123314291457 ],
			"zoom" : 0.65999999999999992
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
**Time**
TSDF lo: 0.1
tsdf lo sm: 0.1
tsdf hi: 2.9
tsdf hi sm: 6.9
SPF_unc: 20.2
SPF_van: 19.1

SPF_unc:
{'recon_to_gt': {'mean': 0.00010914265899911337, 'median': 7.627319465207394e-05, 'rms': 0.00015993955815357666, 'p95': 0.00032197243928846895, 'p99': 0.0005508465259502197, 'hausdorff': 0.0022660181267014634, 'trimmed_hausdorff_99': 0.0005508465259502197, 'count': 150000}, 'gt_to_recon': {'mean': 0.00013150443834547738, 'median': 7.946673193089115e-05, 'rms': 0.0002450472775026421, 'p95': 0.0003871108187492854, 'p99': 0.0010589896629143367, 'hausdorff': 0.0038035704987654895, 'trimmed_hausdorff_99': 0.0010589896629143367, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.9082866666666667, 'recall': 0.88404, 'fscore': 0.8959993283962373}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9863533333333333, 'recall': 0.9697066666666667, 'fscore': 0.977959165896979}, 0.001: {'tau_m': 0.001, 'precision': 0.9987266666666667, 'recall': 0.9890066666666667, 'fscore': 0.9938429013057867}, 0.002: {'tau_m': 0.002, 'precision': 0.99998, 'recall': 0.9976733333333333, 'fscore': 0.9988253349263134}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 7.889799193586102, 'median': 4.906809885681148, 'p90': 17.494339093476313, 'p95': 24.417099158057606, 'max': 89.46873364109462, 'count': 150000}, 'gt_to_recon': {'mean': 8.298727504199325, 'median': 5.0267568128774505, 'p90': 18.282847614501595, 'p95': 25.979514700086188, 'max': 89.98988146708805, 'count': 149999}, 'symmetric': {'mean': 8.094262667343257, 'median': 4.965163412105844, 'p90': 17.90758404233296, 'p95': 25.175355483638715, 'max': 89.98988146708805, 'count': 299999}}, 'recon_topology': {'boundary_edges': 371, 'connected_components_number': 2, 'edges_number': 163063, 'faces_number': 108585, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 136, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 28, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 54415}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}

SPF_van
{'recon_to_gt': {'mean': 0.00014889655793912587, 'median': 7.51717739375865e-05, 'rms': 0.00038111213743055356, 'p95': 0.00046849002934544107, 'p99': 0.0011111956765748174, 'hausdorff': 0.010662294762143965, 'trimmed_hausdorff_99': 0.0011111956765748174, 'count': 150000}, 'gt_to_recon': {'mean': 0.00017728999415212473, 'median': 8.050900027418983e-05, 'rms': 0.0003832395560048757, 'p95': 0.0006270965318794016, 'p99': 0.0018214661163247464, 'hausdorff': 0.005598633950088667, 'trimmed_hausdorff_99': 0.0018214661163247464, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.8587533333333334, 'recall': 0.8283933333333333, 'fscore': 0.8433001710712104}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9553333333333334, 'recall': 0.9322733333333333, 'fscore': 0.9436624767636383}, 0.001: {'tau_m': 0.001, 'precision': 0.9879466666666666, 'recall': 0.97362, 'fscore': 0.9807310145971757}, 0.002: {'tau_m': 0.002, 'precision': 0.9963333333333333, 'recall': 0.99184, 'fscore': 0.994081589130391}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 8.79802372295156, 'median': 5.089607756828072, 'p90': 19.810471171139778, 'p95': 28.836075308964876, 'max': 89.99662178800115, 'count': 149743}, 'gt_to_recon': {'mean': 9.36223561576657, 'median': 5.2828852541645155, 'p90': 21.21117549496689, 'p95': 31.552585309989425, 'max': 89.99415940301478, 'count': 149889}, 'symmetric': {'mean': 9.08026712953747, 'median': 5.1893630878533346, 'p90': 20.489703931209526, 'p95': 30.180425101669798, 'max': 89.99662178800115, 'count': 299632}}, 'recon_topology': {'boundary_edges': 687, 'connected_components_number': 4, 'edges_number': 162321, 'faces_number': 107985, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 627, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 111, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 54108}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}

tsdf_lo
{'recon_to_gt': {'mean': 0.00014584756036953149, 'median': 0.00010870804096343306, 'rms': 0.00020257323919018578, 'p95': 0.00040984875296882627, 'p99': 0.000649489614020064, 'hausdorff': 0.0021316060149987326, 'trimmed_hausdorff_99': 0.000649489614020064, 'count': 150000}, 'gt_to_recon': {'mean': 0.0001431763982790099, 'median': 0.00010530175311265766, 'rms': 0.00021164280326260262, 'p95': 0.00039184997842161255, 'p99': 0.0006387333870475562, 'hausdorff': 0.003357046373094453, 'trimmed_hausdorff_99': 0.0006387333870475562, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.83652, 'recall': 0.8495266666666667, 'fscore': 0.8429731646810883}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9735866666666667, 'recall': 0.9773933333333333, 'fscore': 0.9754862862994211}, 0.001: {'tau_m': 0.001, 'precision': 0.9979333333333333, 'recall': 0.9959733333333334, 'fscore': 0.9969523699983728}, 0.002: {'tau_m': 0.002, 'precision': 0.99998, 'recall': 0.9992666666666666, 'fscore': 0.9996232060742879}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 16.668841699209814, 'median': 13.771683945760834, 'p90': 31.997860615589374, 'p95': 40.270741194216825, 'max': 89.98407838867075, 'count': 150000}, 'gt_to_recon': {'mean': 16.435978067581793, 'median': 13.661505092980274, 'p90': 31.30124041022403, 'p95': 39.19970641634101, 'max': 89.99040263051923, 'count': 150000}, 'symmetric': {'mean': 16.552409883395807, 'median': 13.713692161629922, 'p90': 31.62264068376739, 'p95': 39.74487793204956, 'max': 89.99040263051923, 'count': 300000}}, 'recon_topology': {'boundary_edges': 709, 'connected_components_number': 29, 'edges_number': 277883, 'faces_number': 185019, 'genus': 1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 97, 'unreferenced_vertices': 0, 'vertices_number': 92823}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}

tsdf_lo_smth
{'recon_to_gt': {'mean': 0.00012192553331186725, 'median': 8.802344825382946e-05, 'rms': 0.00017031990979992704, 'p95': 0.00035036766130724093, 'p99': 0.0005443186447852375, 'hausdorff': 0.0017031838008306766, 'trimmed_hausdorff_99': 0.0005443186447852375, 'count': 150000}, 'gt_to_recon': {'mean': 0.0001390357230419276, 'median': 9.190477064285885e-05, 'rms': 0.00023068343627159128, 'p95': 0.00039931947358381277, 'p99': 0.0008008796738580871, 'hausdorff': 0.0039304687129247725, 'trimmed_hausdorff_99': 0.0008008796738580871, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.8812266666666667, 'recall': 0.86, 'fscore': 0.8704839500122519}, 0.0005: {'tau_m': 0.0005, 'precision': 0.98592, 'recall': 0.9724733333333333, 'fscore': 0.9791505030994795}, 0.001: {'tau_m': 0.001, 'precision': 0.9992466666666666, 'recall': 0.9934333333333333, 'fscore': 0.9963315202530147}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 0.9984333333333333, 'fscore': 0.9992160525745167}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 8.739440422223732, 'median': 6.307114235810289, 'p90': 18.092360734045663, 'p95': 23.859923583759585, 'max': 89.79548020211828, 'count': 150000}, 'gt_to_recon': {'mean': 9.062761457698246, 'median': 6.398194279411641, 'p90': 18.691124399604142, 'p95': 24.852866308030872, 'max': 90.0, 'count': 150000}, 'symmetric': {'mean': 8.901100939960989, 'median': 6.35360791224853, 'p90': 18.39534826346344, 'p95': 24.328276429679526, 'max': 90.0, 'count': 300000}}, 'recon_topology': {'boundary_edges': 709, 'connected_components_number': 29, 'edges_number': 277883, 'faces_number': 185019, 'genus': 1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 97, 'unreferenced_vertices': 0, 'vertices_number': 92823}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}

tsdf_hi
{'recon_to_gt': {'mean': 0.00012789031965644606, 'median': 0.00010039623094934558, 'rms': 0.000170801205739938, 'p95': 0.0003329227005326922, 'p99': 0.000519781291191156, 'hausdorff': 0.0016649825674740504, 'trimmed_hausdorff_99': 0.000519781291191156, 'count': 150000}, 'gt_to_recon': {'mean': 0.00010633786885435142, 'median': 8.310597065884495e-05, 'rms': 0.00015288091911224824, 'p95': 0.0002705866111716163, 'p99': 0.0004022824973883609, 'hausdorff': 0.002878820697065802, 'trimmed_hausdorff_99': 0.0004022824973883609, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.8798133333333333, 'recall': 0.9341266666666667, 'fscore': 0.9061568699687482}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9887533333333334, 'recall': 0.9946, 'fscore': 0.9916680490616901}, 0.001: {'tau_m': 0.001, 'precision': 0.9994266666666667, 'recall': 0.9982333333333333, 'fscore': 0.998829643571868}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 0.99966, 'fscore': 0.9998299710950862}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 30.393910315528945, 'median': 28.014321353804924, 'p90': 54.49103316963147, 'p95': 64.11515910618901, 'max': 89.98946599896951, 'count': 150000}, 'gt_to_recon': {'mean': 30.351246614221857, 'median': 28.00054389251798, 'p90': 54.38465023865246, 'p95': 64.0454452749796, 'max': 90.0, 'count': 150000}, 'symmetric': {'mean': 30.372578464875403, 'median': 28.007284986031102, 'p90': 54.441698917434614, 'p95': 64.07289197420509, 'max': 90.0, 'count': 300000}}, 'recon_topology': {'boundary_edges': 2726, 'connected_components_number': 198, 'edges_number': 5027641, 'faces_number': 3350852, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 2, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 1, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 1676899}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}


tsdf_hi_smth
{'recon_to_gt': {'mean': 8.049069047179969e-05, 'median': 5.5762589646721206e-05, 'rms': 0.00011473763557311354, 'p95': 0.00024026878040312583, 'p99': 0.0003808234144328887, 'hausdorff': 0.001138069837089055, 'trimmed_hausdorff_99': 0.0003808234144328887, 'count': 150000}, 'gt_to_recon': {'mean': 9.291054075361545e-05, 'median': 5.7855548451168914e-05, 'rms': 0.0001711773695352482, 'p95': 0.00027271692900548386, 'p99': 0.0005355669885918418, 'hausdorff': 0.003695760021822287, 'trimmed_hausdorff_99': 0.0005355669885918418, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.9555733333333334, 'recall': 0.9387533333333333, 'fscore': 0.9470886597289919}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9968933333333333, 'recall': 0.9886733333333333, 'fscore': 0.992766318442538}, 0.001: {'tau_m': 0.001, 'precision': 0.9999733333333334, 'recall': 0.9960933333333334, 'fscore': 0.9980295623170012}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 0.9989866666666667, 'fscore': 0.9994930764920893}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 6.953713930896842, 'median': 4.920244480238601, 'p90': 14.792985980610204, 'p95': 19.404745577479467, 'max': 88.97051012498495, 'count': 150000}, 'gt_to_recon': {'mean': 7.242666962200201, 'median': 4.958353788583402, 'p90': 15.079537604405392, 'p95': 20.198480262473975, 'max': 90.0, 'count': 150000}, 'symmetric': {'mean': 7.098190446548521, 'median': 4.938419763422155, 'p90': 14.933462045036537, 'p95': 19.785474377839225, 'max': 90.0, 'count': 300000}}, 'recon_topology': {'boundary_edges': 2726, 'connected_components_number': 198, 'edges_number': 5027641, 'faces_number': 3350852, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 2, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 1, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 1676899}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}


SPF_unc
{'mse': 0.0010487220924084463, 'iou': 0.9917930027406292}

SPF_van
{'mse': 0.0012882138162101113, 'iou': 0.98844842029093}

tsdf_lo
{'mse': 0.0019712958440211096, 'iou': 0.9892930066541767}

tsdf_lo_smth
{'mse': 0.003069678776389067, 'iou': 0.9943452015882603}

tsdf_hi
{'mse': 0.0008405244640727913, 'iou': 0.9899088847304964}

tsdf_hi_smth
{'mse': 0.0024608832466103733, 'iou': 0.9963292905647317}


### Mustard
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.047320038080215454, 0.030810520052909851, 0.9113733172416687 ],
			"boundingbox_min" : [ -0.049816854298114777, -0.035761035978794098, 0.72007954120635986 ],
			"field_of_view" : 60.0,
			"front" : [ -0.033191651460816318, 0.99835792013933211, 0.046688088077911916 ],
			"lookat" : [ -0.0031176054006561774, 0.0024845070841953989, 0.81448694588758952 ],
			"up" : [ -0.0062168156497926993, -0.046919158173194017, 0.99887934396477329 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

TSDF 1/1000, 4: 0.1s
SPF_unc: 20.3s
SPF_vanla: 19.2

SPF_unc
{'recon_to_gt': {'mean': 0.00011974408997313754, 'median': 5.6613328114237665e-05, 'rms': 0.00018579022725687298, 'p95': 0.0004114770991563444, 'p99': 0.0006063744106078642, 'hausdorff': 0.00116641498371306, 'trimmed_hausdorff_99': 0.0006063744106078642, 'count': 150000}, 'gt_to_recon': {'mean': 0.00014647816048920555, 'median': 5.9994470371553185e-05, 'rms': 0.0003624366650702694, 'p95': 0.00044708575498421594, 'p99': 0.0007868501748846536, 'hausdorff': 0.007281924130989923, 'trimmed_hausdorff_99': 0.0007868501748846536, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.8266666666666667, 'recall': 0.8114533333333334, 'fscore': 0.8189893563624426}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9770933333333334, 'recall': 0.96464, 'fscore': 0.9708267318547003}, 0.001: {'tau_m': 0.001, 'precision': 0.99982, 'recall': 0.9932333333333333, 'fscore': 0.9965157828189912}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 0.99524, 'fscore': 0.9976143220865661}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 3.2242785925487687, 'median': 1.7316668201630299, 'p90': 7.181517544878193, 'p95': 10.225247157023533, 'max': 70.13532480598427, 'count': 150000}, 'gt_to_recon': {'mean': 3.317032454245205, 'median': 1.7634762438161518, 'p90': 7.381433845132793, 'p95': 10.556415947532408, 'max': 81.98314595485192, 'count': 149676}, 'symmetric': {'mean': 3.270605382159134, 'median': 1.7479083762127816, 'p90': 7.2797567245574974, 'p95': 10.3924657181, 'max': 81.98314595485192, 'count': 299676}}, 'recon_topology': {'boundary_edges': 109, 'connected_components_number': 1, 'edges_number': 130730, 'faces_number': 87117, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 44, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 7, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 43596}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

SPF_van
{'recon_to_gt': {'mean': 0.00012152714471638837, 'median': 5.126935554351786e-05, 'rms': 0.00020821884695920734, 'p95': 0.00042621328547394173, 'p99': 0.0008581532934595508, 'hausdorff': 0.0015977471464047044, 'trimmed_hausdorff_99': 0.0008581532934595508, 'count': 150000}, 'gt_to_recon': {'mean': 0.00013180780102948393, 'median': 5.428097246296809e-05, 'rms': 0.00023084414225687567, 'p95': 0.0004659280381626975, 'p99': 0.0010141627351998546, 'hausdorff': 0.0015866345643638319, 'trimmed_hausdorff_99': 0.0010141627351998546, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.83514, 'recall': 0.8224733333333333, 'fscore': 0.8287582704450575}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9668, 'recall': 0.9570533333333333, 'fscore': 0.9619019772817056}, 0.001: {'tau_m': 0.001, 'precision': 0.9941266666666667, 'recall': 0.9895, 'fscore': 0.9918079376495578}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 3.4650248808284996, 'median': 1.6772352342004986, 'p90': 7.555504092983414, 'p95': 11.966368567319037, 'max': 74.87291239235766, 'count': 150000}, 'gt_to_recon': {'mean': 3.5170849950959084, 'median': 1.7163364468765765, 'p90': 7.822396607639175, 'p95': 12.389660159055813, 'max': 76.46114849051105, 'count': 150000}, 'symmetric': {'mean': 3.491054937962204, 'median': 1.6957810706838068, 'p90': 7.683455433478356, 'p95': 12.184137901354879, 'max': 76.46114849051105, 'count': 300000}}, 'recon_topology': {'boundary_edges': 56, 'connected_components_number': 1, 'edges_number': 129880, 'faces_number': 86568, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 70, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 13, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 43286}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

TSDF
{'recon_to_gt': {'mean': 0.0001192379570269229, 'median': 9.335267320731595e-05, 'rms': 0.0001586808891268807, 'p95': 0.00031922678787943686, 'p99': 0.0004822000877227339, 'hausdorff': 0.0015365678545202488, 'trimmed_hausdorff_99': 0.0004822000877227339, 'count': 150000}, 'gt_to_recon': {'mean': 0.00011126066121417222, 'median': 8.848958154148573e-05, 'rms': 0.00014544401574468117, 'p95': 0.00029397160977503165, 'p99': 0.00042858129434483944, 'hausdorff': 0.0009262724383125022, 'trimmed_hausdorff_99': 0.00042858129434483944, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.8977333333333334, 'recall': 0.9156866666666666, 'fscore': 0.9066211286470377}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9917066666666666, 'recall': 0.9961066666666667, 'fscore': 0.9939017969941286}, 0.001: {'tau_m': 0.001, 'precision': 0.99976, 'recall': 1.0, 'fscore': 0.9998799855982717}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 13.402483014118804, 'median': 11.219240215569153, 'p90': 25.05098982534493, 'p95': 30.86652878320172, 'max': 89.91855823407695, 'count': 150000}, 'gt_to_recon': {'mean': 13.114595157757401, 'median': 11.070032148375205, 'p90': 24.32748701309965, 'p95': 29.85783780855374, 'max': 90.0, 'count': 150000}, 'symmetric': {'mean': 13.258539085938104, 'median': 11.144454858451628, 'p90': 24.68511181803749, 'p95': 30.377528194604146, 'max': 90.0, 'count': 300000}}, 'recon_topology': {'boundary_edges': 77, 'connected_components_number': 9, 'edges_number': 205180, 'faces_number': 136761, 'genus': 0, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 15, 'unreferenced_vertices': 0, 'vertices_number': 68422}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

SPF_unc
{'mse': 0.0014879863013830294, 'iou': 0.9819315346695654}

SPF_van
{'mse': 0.001781667522806911, 'iou': 0.978447900854623}

TSDF
{'mse': 0.0019712958440211096, 'iou': 0.9892930066541767}