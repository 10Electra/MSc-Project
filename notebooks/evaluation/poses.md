
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

h_alpha=5.0, r_alpha=2.0,
nrm_shift_iters=1, nrm_smth_iters=1,
sigma_theta=0.15,
normal_diff_thresh=20,
ball_radius_percentiles=[1,5,15,50,85,95,99],
bilateral_weight_update=False,
shift_all=False,

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

TSDF_lo: 0.1
tsdf_lo_sm: 0.1
tsdf_hi: 1.9
tsdf_hi_sm: 5.3
SPF_unc: 19.6
SPF_van: 19.1

SPF_unc
{'recon_to_gt': {'mean': 3.758296748024782e-05, 'median': 1.809669733475114e-05, 'rms': 6.144406595006659e-05, 'p95': 0.00012769634867826059, 'p99': 0.00021394576850288313, 'hausdorff': 0.0006836954628253178, 'trimmed_hausdorff_99': 0.00021394576850288313, 'count': 150000}, 'gt_to_recon': {'mean': 3.861742964496185e-05, 'median': 1.831259645274086e-05, 'rms': 6.546736262978642e-05, 'p95': 0.0001294676926037874, 'p99': 0.0002275486556966132, 'hausdorff': 0.0009972359392712016, 'trimmed_hausdorff_99': 0.0002275486556966132, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.9942733333333333, 'recall': 0.9926533333333334, 'fscore': 0.9934626729164081}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9994666666666666, 'recall': 0.9985866666666666, 'fscore': 0.9990264728780457}, 0.001: {'tau_m': 0.001, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 2.234688883386588, 'median': 1.2353497865250527, 'p90': 4.703347022444625, 'p95': 6.689076587851943, 'max': 90.0, 'count': 150000}, 'gt_to_recon': {'mean': 2.2759171416386543, 'median': 1.2563707202585808, 'p90': 4.7655352972633604, 'p95': 6.817419618448588, 'max': 88.6646177189659, 'count': 150000}, 'symmetric': {'mean': 2.255303012512621, 'median': 1.245471404028839, 'p90': 4.733250161761601, 'p95': 6.760490431845966, 'max': 90.0, 'count': 300000}}, 'recon_topology': {'boundary_edges': 57, 'connected_components_number': 1, 'edges_number': 174375, 'faces_number': 116231, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 44, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 8, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 58119}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

SPF_van
{'recon_to_gt': {'mean': 4.0384852380717256e-05, 'median': 1.7132701139285084e-05, 'rms': 7.524295606832672e-05, 'p95': 0.0001395806485170233, 'p99': 0.00035224874428021444, 'hausdorff': 0.0007376469518846074, 'trimmed_hausdorff_99': 0.00035224874428021444, 'count': 150000}, 'gt_to_recon': {'mean': 4.270451252663487e-05, 'median': 1.73756302581686e-05, 'rms': 8.345590710805457e-05, 'p95': 0.00014688153927260158, 'p99': 0.00038730854653130036, 'hausdorff': 0.0009785458782574086, 'trimmed_hausdorff_99': 0.00038730854653130036, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.9809066666666667, 'recall': 0.97676, 'fscore': 0.9788289416652477}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9977666666666667, 'recall': 0.99582, 'fscore': 0.9967923829078579}, 0.001: {'tau_m': 0.001, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 2.2885975683955855, 'median': 1.1936513799932142, 'p90': 4.797657887764692, 'p95': 7.218741531532033, 'max': 67.04380587302663, 'count': 150000}, 'gt_to_recon': {'mean': 2.342966266144733, 'median': 1.219399604903899, 'p90': 4.859722859331908, 'p95': 7.4399067264634375, 'max': 86.54389293381891, 'count': 150000}, 'symmetric': {'mean': 2.3157819172701593, 'median': 1.2066205679520516, 'p90': 4.830166158113042, 'p95': 7.328302740384979, 'max': 86.54389293381891, 'count': 300000}}, 'recon_topology': {'boundary_edges': 20, 'connected_components_number': 1, 'edges_number': 174154, 'faces_number': 116096, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 38, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 6, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 58049}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

tsdf_lo
{'recon_to_gt': {'mean': 0.00011092313379871096, 'median': 8.34117831873607e-05, 'rms': 0.00015213555092778548, 'p95': 0.0003087966048076047, 'p99': 0.0004935521555266538, 'hausdorff': 0.0016186546227601184, 'trimmed_hausdorff_99': 0.0004935521555266538, 'count': 150000}, 'gt_to_recon': {'mean': 0.00010395317678521375, 'median': 8.059037894918297e-05, 'rms': 0.00013830723774120756, 'p95': 0.00028419230635734195, 'p99': 0.00042503903324442956, 'hausdorff': 0.0009138957998547914, 'trimmed_hausdorff_99': 0.00042503903324442956, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.9116466666666667, 'recall': 0.92692, 'fscore': 0.9192198940478997}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9905266666666667, 'recall': 0.9957066666666666, 'fscore': 0.9931099120725438}, 0.001: {'tau_m': 0.001, 'precision': 0.9997466666666667, 'recall': 1.0, 'fscore': 0.9998733172868562}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 12.768408874054042, 'median': 10.655414904291158, 'p90': 23.985159971911983, 'p95': 30.16624558726414, 'max': 89.86556079815325, 'count': 150000}, 'gt_to_recon': {'mean': 12.50207715601533, 'median': 10.487567758937285, 'p90': 23.431252721935298, 'p95': 29.301401950732462, 'max': 89.92876701946884, 'count': 150000}, 'symmetric': {'mean': 12.635243015034685, 'median': 10.570326522525173, 'p90': 23.716462113877174, 'p95': 29.753046834319743, 'max': 89.92876701946884, 'count': 300000}}, 'recon_topology': {'boundary_edges': 121, 'connected_components_number': 8, 'edges_number': 202754, 'faces_number': 135129, 'genus': 0, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 21, 'unreferenced_vertices': 0, 'vertices_number': 67620}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

tsdf_lo_sm
{'recon_to_gt': {'mean': 7.352038227595891e-05, 'median': 5.548410136772125e-05, 'rms': 0.0001001215368255377, 'p95': 0.00020496659091981672, 'p99': 0.00032492141235117766, 'hausdorff': 0.000893037585018811, 'trimmed_hausdorff_99': 0.00032492141235117766, 'count': 150000}, 'gt_to_recon': {'mean': 7.424136198636982e-05, 'median': 5.545155079813083e-05, 'rms': 0.00010249635483444255, 'p95': 0.00020629872715625846, 'p99': 0.0003357382115314736, 'hausdorff': 0.0010869523918122827, 'trimmed_hausdorff_99': 0.0003357382115314736, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.9740066666666667, 'recall': 0.97272, 'fscore': 0.973362908129544}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9990733333333334, 'recall': 0.99804, 'fscore': 0.9985563993363754}, 0.001: {'tau_m': 0.001, 'precision': 1.0, 'recall': 0.9999933333333333, 'fscore': 0.9999966666555555}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 4.060127018015723, 'median': 3.0830449742461368, 'p90': 7.691909769658968, 'p95': 10.263573001455052, 'max': 78.51973969119386, 'count': 150000}, 'gt_to_recon': {'mean': 4.04182170861267, 'median': 3.067179837866317, 'p90': 7.621456103127095, 'p95': 10.141276819957676, 'max': 66.53448204268558, 'count': 150000}, 'symmetric': {'mean': 4.050974363314197, 'median': 3.0741676575160275, 'p90': 7.6565786895667305, 'p95': 10.204652256872286, 'max': 78.51973969119386, 'count': 300000}}, 'recon_topology': {'boundary_edges': 121, 'connected_components_number': 8, 'edges_number': 202754, 'faces_number': 135129, 'genus': 0, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 21, 'unreferenced_vertices': 0, 'vertices_number': 67620}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

tsdf_hi
{'recon_to_gt': {'mean': 0.00010193201528187906, 'median': 8.275058982428786e-05, 'rms': 0.00013262627262479107, 'p95': 0.00025909540338261977, 'p99': 0.0003713078271232459, 'hausdorff': 0.0011474472911872091, 'trimmed_hausdorff_99': 0.0003713078271232459, 'count': 150000}, 'gt_to_recon': {'mean': 8.557235925895042e-05, 'median': 7.093309794741279e-05, 'rms': 0.00010867443180881372, 'p95': 0.00021496313989230014, 'p99': 0.0002906258349043323, 'hausdorff': 0.0005656510886763132, 'trimmed_hausdorff_99': 0.0002906258349043323, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.9428666666666666, 'recall': 0.9757466666666667, 'fscore': 0.9590249281888771}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9971666666666666, 'recall': 0.99996, 'fscore': 0.9985613798490499}, 0.001: {'tau_m': 0.001, 'precision': 0.9999333333333333, 'recall': 1.0, 'fscore': 0.9999666655555185}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 26.443642618491833, 'median': 23.862697636909637, 'p90': 48.82843437612763, 'p95': 57.37530436420989, 'max': 89.9838086592937, 'count': 150000}, 'gt_to_recon': {'mean': 26.391854546463108, 'median': 23.807989551591348, 'p90': 48.846856268315285, 'p95': 57.30707633164521, 'max': 90.0, 'count': 150000}, 'symmetric': {'mean': 26.41774858247747, 'median': 23.83664437534569, 'p90': 48.83842973306263, 'p95': 57.33932715019709, 'max': 90.0, 'count': 300000}}, 'recon_topology': {'boundary_edges': 80, 'connected_components_number': 74, 'edges_number': 3677362, 'faces_number': 2451548, 'genus': 20, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 10, 'unreferenced_vertices': 0, 'vertices_number': 1225912}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

tsdf_hi_sm
{'recon_to_gt': {'mean': 4.0218280992706344e-05, 'median': 2.6730629010148638e-05, 'rms': 5.899231483567688e-05, 'p95': 0.00012463476736719695, 'p99': 0.00020388289148606987, 'hausdorff': 0.0005864445303115362, 'trimmed_hausdorff_99': 0.00020388289148606987, 'count': 150000}, 'gt_to_recon': {'mean': 4.0939152457329566e-05, 'median': 2.681771082832171e-05, 'rms': 6.124292454943795e-05, 'p95': 0.0001261646330278954, 'p99': 0.00021268393505686055, 'hausdorff': 0.0006705042776703474, 'trimmed_hausdorff_99': 0.00021268393505686055, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.9956266666666667, 'recall': 0.9939933333333333, 'fscore': 0.9948093295760553}, 0.0005: {'tau_m': 0.0005, 'precision': 0.99996, 'recall': 0.9997666666666667, 'fscore': 0.9998633239876116}, 0.001: {'tau_m': 0.001, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}, 0.002: {'tau_m': 0.002, 'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 2.75510330282276, 'median': 1.9508772148832578, 'p90': 5.523291614204077, 'p95': 7.276777702318361, 'max': 77.87633634379499, 'count': 150000}, 'gt_to_recon': {'mean': 2.7417025683249188, 'median': 1.9354658496375081, 'p90': 5.515429089245041, 'p95': 7.21571304305591, 'max': 90.0, 'count': 150000}, 'symmetric': {'mean': 2.7484029355738393, 'median': 1.9430003638198765, 'p90': 5.520034252009598, 'p95': 7.250267417756631, 'max': 90.0, 'count': 300000}}, 'recon_topology': {'boundary_edges': 80, 'connected_components_number': 74, 'edges_number': 3677362, 'faces_number': 2451548, 'genus': 20, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 10, 'unreferenced_vertices': 0, 'vertices_number': 1225912}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}


SPF_unc
{'mse': 0.0007310324048725676, 'iou': 0.9962618713831751}

SPF_van
{'mse': 0.0008032237257825545, 'iou': 0.9952176150496602}

tsdf_lo
{'mse': 0.002475293277673485, 'iou': 0.9912400398422077}

tsdf_lo_sm
{'mse': 0.0038987847163544946, 'iou': 0.9964477576572089}

tsdf_hi
{'mse': 0.0013685171037203126, 'iou': 0.9911962426501753}

tsdf_hi_sm
{'mse': 0.003271854596404237, 'iou': 0.9983063692830487}


### Scene
cam_offset=obj_centre
look_at = obj_centre
width: int = 360
height: int = 240
fov: float = 70.0
k: float = 3
a = 45
c = int(360/10)
rs = (0.45, 0.45)
ls = (50, 80)
for i in range(360//c):
    r = rs[0] if i%2==0 else rs[1]
    l = ls[0] if i%2==0 else ls[1]
    ccs.append(cam_offset + polar2cartesian(r=r, lat=l, long=a+i*c))

{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.89999419450759888, 0.44999271631240845, 0.97084990505266999 ],
			"boundingbox_min" : [ -0.89998292922975187, -0.45000013709068298, 0.71999996900552343 ],
			"field_of_view" : 60.0,
			"front" : [ 0.052072202370523381, 0.89322812194070089, 0.44657810953356403 ],
			"lookat" : [ 5.6326389235028529e-06, -3.7103891372680664e-06, 0.84542493702909671 ],
			"up" : [ -0.015416323196240082, -0.44641250866415461, 0.89469447806906921 ],
			"zoom" : 0.11999999999999962
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

tsdf_lo: 3.1
tsdf_lo_sm: 3.1
tsdf_hi: 4.2
tsdf_hi_sm: 5.0
sdf_unc: 142.0
sdf_van: 95.4


tsdf_lo
438315

tsdf_hi


sdf_unc
453308

sdf_van
431616

## 2 Noise

### Drill
width: int = 360
height: int = 240
fov: float = 70.0
k: float = 5

a = 45
c = 45
rs = (0.35, 0.5, 1.0)
ls = (90,90)
for i in range(360//c):
    r = rs[i%len(rs)]
    l = ls[i%len(ls)]
        constant_uncertainty    =2e-5,
        linear_uncertainty      =1e-5,      # rate of uncertainty increase with depth
        quadrt_uncertainty      =3e-3,      # quadratic uncertainty coefficient
        constant_perlin_sigma   =2e-5,      # constant perlin noise term
        linear_perlin_sigma     =1e-5,      # linear depth term
        quadrt_perlin_sigma     =3e-3,      # quadratic depth term
        perlin_octaves          =5,
        seg_scale_std           =1e-8,#4,      # std of per-segment scale noise
        rot_std                 =1e-8,#4,      # std of global rotation noise
        trn_std                 =5e-8,#3,      # std of global translation noise
        grazing_lambda          =2.0,       # sigma multiplier at grazing angles; 0 disables

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

### Must
cam_offset=obj_centre
look_at = obj_centre
width: int = 360
height: int = 240
fov: float = 70.0
k: float = 5
max_normal_angle_deg = None
N = 1

scans = []
ccs = []
a = 45
c = 60
rs = (0.35, 1)
ls = (50, 80)
for i in range(360//c):
    r = rs[0] if i%2==0 else rs[1]
    l = ls[0] if i%2==0 else ls[1]
    ccs.append(cam_offset + polar2cartesian(r=r, lat=l, long=a+i*c))
for cc in ccs:
    object_meshes, object_weights = virtual_mesh_scan(
        gt_mesh_list,
        cc,
        look_at,
        k=k,
        max_normal_angle_deg=max_normal_angle_deg,
        width_px=width,
        height_px=height,
        fov=fov,
        constant_uncertainty    =2e-4,
        linear_uncertainty      =1e-3,      # rate of uncertainty increase with depth
        quadrt_uncertainty      =3e-4,      # quadratic uncertainty coefficient
        constant_perlin_sigma   =2e-4,      # constant perlin noise term
        linear_perlin_sigma     =1e-3,      # linear depth term
        quadrt_perlin_sigma     =3e-4,      # quadratic depth term
        perlin_octaves          =3,
        seg_scale_std           =1e-4,      # std of per-segment scale noise
        rot_std                 =1e-4,      # std of global rotation noise
        trn_std                 =1e-3,      # std of global translation noise
        grazing_lambda          =1.0,       # sigma multiplier at grazing angles; 0 disables
        seed                    =None,
        include_depth_image     =False,
    )
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

h_alpha         = 3,
r_alpha         = 3,
nrm_shift_iters = 2,
nrm_smth_iters  = 1,
sigma_theta = None,
normal_diff_thresh = None,
tau_max = 500_000,
shift_all = False,
ball_radius_percentiles = [10, 50, 90],
bilateral_weight_update = False,
resp_frac = 0.3,

unc_nrm
{'recon_to_gt': {'mean': 0.0011748596543343438, 'median': 0.0011631427655090648, 'rms': 0.0013672145291034974, 'p95': 0.0022245632107681028, 'p99': 0.0030069603021518876, 'hausdorff': 0.003883252547403848, 'trimmed_hausdorff_99': 0.0030069603021518876, 'count': 150000}, 'gt_to_recon': {'mean': 0.002135301580451509, 'median': 0.0013156494287154333, 'rms': 0.004215398912667701, 'p95': 0.009413180321591915, 'p99': 0.02121413310553163, 'hausdorff': 0.027464467404210928, 'trimmed_hausdorff_99': 0.02121413310553163, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.10322666666666666, 'recall': 0.09006, 'fscore': 0.09619487738419619}, 0.0005: {'tau_m': 0.0005, 'precision': 0.21186, 'recall': 0.18522, 'fscore': 0.19764636446056208}, 0.001: {'tau_m': 0.001, 'precision': 0.42586666666666667, 'recall': 0.37522666666666665, 'fscore': 0.3989460981103603}, 0.002: {'tau_m': 0.002, 'precision': 0.88996, 'recall': 0.79812, 'fscore': 0.841541722193261}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 6.660868825333859, 'median': 3.3708631955819044, 'p90': 14.998074281050211, 'p95': 24.560564021714843, 'max': 89.96183578268177, 'count': 150000}, 'gt_to_recon': {'mean': 7.5397895475946655, 'median': 3.5435651943775657, 'p90': 17.88930237528849, 'p95': 31.74782204121157, 'max': 89.85591850023884, 'count': 138453}, 'symmetric': {'mean': 7.082737243270838, 'median': 3.4516673284370056, 'p90': 16.253869309736213, 'p95': 28.099415185740735, 'max': 89.96183578268177, 'count': 288453}}, 'recon_topology': {'boundary_edges': 125, 'connected_components_number': 1, 'edges_number': 33757, 'faces_number': 22463, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 23, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 5, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 11277}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

van_nrm
{'recon_to_gt': {'mean': 0.001178799960257924, 'median': 0.0011576086299102326, 'rms': 0.0013762271825575802, 'p95': 0.0022502972346217015, 'p99': 0.0030398821181350237, 'hausdorff': 0.00465158800219722, 'trimmed_hausdorff_99': 0.0030398821181350237, 'count': 150000}, 'gt_to_recon': {'mean': 0.0021430173911966147, 'median': 0.0013114815904133603, 'rms': 0.004221199367903822, 'p95': 0.009483380565997556, 'p99': 0.0212223999807649, 'hausdorff': 0.02743222278461104, 'trimmed_hausdorff_99': 0.0212223999807649, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.1046, 'recall': 0.08928666666666667, 'fscore': 0.09633860330777431}, 0.0005: {'tau_m': 0.0005, 'precision': 0.21286666666666668, 'recall': 0.18507333333333334, 'fscore': 0.1979994147638114}, 0.001: {'tau_m': 0.001, 'precision': 0.4286466666666667, 'recall': 0.37556, 'fscore': 0.4003511754026743}, 0.002: {'tau_m': 0.002, 'precision': 0.88414, 'recall': 0.79354, 'fscore': 0.8363936574316915}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 6.791995957066899, 'median': 3.4174038237889315, 'p90': 15.417970673879488, 'p95': 24.7928692571228, 'max': 89.91572253889905, 'count': 149977}, 'gt_to_recon': {'mean': 7.678097451258533, 'median': 3.5792727451648108, 'p90': 18.434697903072788, 'p95': 31.881446079730953, 'max': 89.9020356462353, 'count': 138328}, 'symmetric': {'mean': 7.217145182673602, 'median': 3.492852707330817, 'p90': 16.706931864292198, 'p95': 28.37188871740143, 'max': 89.91572253889905, 'count': 288305}}, 'recon_topology': {'boundary_edges': 168, 'connected_components_number': 1, 'edges_number': 31932, 'faces_number': 21232, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 63, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 11, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 10672}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

unc_nonrm
{'recon_to_gt': {'mean': 0.001282953287085322, 'median': 0.0011964765317182269, 'rms': 0.0015476564719400874, 'p95': 0.0028249675401110577, 'p99': 0.003843470795025093, 'hausdorff': 0.006353819102549737, 'trimmed_hausdorff_99': 0.003843470795025093, 'count': 150000}, 'gt_to_recon': {'mean': 0.0022959903451740932, 'median': 0.0014460994683946605, 'rms': 0.004252110199480634, 'p95': 0.00885398617256253, 'p99': 0.020942719441454892, 'hausdorff': 0.02739612191941918, 'trimmed_hausdorff_99': 0.020942719441454892, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.10442666666666667, 'recall': 0.08684666666666667, 'fscore': 0.09482877464477826}, 0.0005: {'tau_m': 0.0005, 'precision': 0.21724666666666667, 'recall': 0.18261333333333332, 'fscore': 0.19843014032689218}, 0.001: {'tau_m': 0.001, 'precision': 0.4248733333333333, 'recall': 0.3602, 'fscore': 0.3898728135800478}, 0.002: {'tau_m': 0.002, 'precision': 0.8238333333333333, 'recall': 0.7129866666666667, 'fscore': 0.764412464989032}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 7.360824581290267, 'median': 3.9871063111627105, 'p90': 16.791393959662376, 'p95': 25.446422096355402, 'max': 89.77603578691746, 'count': 148916}, 'gt_to_recon': {'mean': 8.357097119557665, 'median': 4.244476808414578, 'p90': 19.694543913447536, 'p95': 32.16491360287621, 'max': 88.5499659844471, 'count': 135227}, 'symmetric': {'mean': 7.834962415170692, 'median': 4.105021203810804, 'p90': 17.994157372840657, 'p95': 28.342742221838428, 'max': 89.77603578691746, 'count': 284143}}, 'recon_topology': {'boundary_edges': 120, 'connected_components_number': 1, 'edges_number': 33075, 'faces_number': 22010, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 42, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 8, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 11052}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

tsdf lo
{'recon_to_gt': {'mean': 0.0008608558495049874, 'median': 0.0006043628742518226, 'rms': 0.0012219968135856014, 'p95': 0.003094961404851794, 'p99': 0.003831464758433489, 'hausdorff': 0.005013812281476543, 'trimmed_hausdorff_99': 0.003831464758433489, 'count': 150000}, 'gt_to_recon': {'mean': 0.0015286045352178679, 'median': 0.0005717482220118641, 'rms': 0.003818723227303796, 'p95': 0.008934345982735218, 'p99': 0.01971476026875042, 'hausdorff': 0.02533332042935994, 'trimmed_hausdorff_99': 0.01971476026875042, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.2275, 'recall': 0.2343, 'fscore': 0.2308499350368125}, 0.0005: {'tau_m': 0.0005, 'precision': 0.42911333333333335, 'recall': 0.44694666666666666, 'fscore': 0.4378484897279993}, 0.001: {'tau_m': 0.001, 'precision': 0.7161666666666666, 'recall': 0.7410533333333333, 'fscore': 0.728397490503226}, 0.002: {'tau_m': 0.002, 'precision': 0.9024333333333333, 'recall': 0.9043266666666666, 'fscore': 0.9033790079725278}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 20.604885914288054, 'median': 13.947188450885248, 'p90': 51.13933163446602, 'p95': 68.84294834930868, 'max': 89.99366266665567, 'count': 149183}, 'gt_to_recon': {'mean': 18.066335679600524, 'median': 13.009937688769462, 'p90': 38.52568784938259, 'p95': 55.080024298390455, 'max': 90.0, 'count': 138626}, 'symmetric': {'mean': 19.382168539800826, 'median': 13.489026509754652, 'p90': 44.19016690688082, 'p95': 63.66340495673789, 'max': 90.0, 'count': 287809}}, 'recon_topology': {'boundary_edges': 392, 'connected_components_number': 22, 'edges_number': 42478, 'faces_number': 28188, 'genus': 38, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 21, 'unreferenced_vertices': 0, 'vertices_number': 14237}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

tsdf lo sm
{'recon_to_gt': {'mean': 0.0007664869548996977, 'median': 0.0005716006305281456, 'rms': 0.0010404162534423853, 'p95': 0.0022915422971915814, 'p99': 0.0033780913539373583, 'hausdorff': 0.004345794558647351, 'trimmed_hausdorff_99': 0.0033780913539373583, 'count': 150000}, 'gt_to_recon': {'mean': 0.0017054917661442305, 'median': 0.0006474463481028238, 'rms': 0.00404359916249014, 'p95': 0.00973474015612102, 'p99': 0.02040833252659408, 'hausdorff': 0.025847897495749936, 'trimmed_hausdorff_99': 0.02040833252659408, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.21775333333333333, 'recall': 0.19558666666666666, 'fscore': 0.20607562114589548}, 0.0005: {'tau_m': 0.0005, 'precision': 0.44602, 'recall': 0.40166666666666667, 'fscore': 0.4226829882110528}, 0.001: {'tau_m': 0.001, 'precision': 0.7451133333333333, 'recall': 0.6817733333333333, 'fscore': 0.7120374909165097}, 0.002: {'tau_m': 0.002, 'precision': 0.9370333333333334, 'recall': 0.8799933333333333, 'fscore': 0.9076180350805101}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 10.311537910825484, 'median': 5.899005236023908, 'p90': 24.06673938696735, 'p95': 35.86525402325274, 'max': 89.96465407869483, 'count': 149908}, 'gt_to_recon': {'mean': 10.442947100152864, 'median': 5.946361436909419, 'p90': 24.0137165025264, 'p95': 35.25148492138951, 'max': 90.0, 'count': 137353}, 'symmetric': {'mean': 10.374370827168752, 'median': 5.920260090203635, 'p90': 24.046300894321483, 'p95': 35.59630766305795, 'max': 90.0, 'count': 287261}}, 'recon_topology': {'boundary_edges': 392, 'connected_components_number': 22, 'edges_number': 42478, 'faces_number': 28188, 'genus': 38, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 21, 'unreferenced_vertices': 0, 'vertices_number': 14237}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

tsdf hi
{'recon_to_gt': {'mean': 0.0011834525378313667, 'median': 0.0007888486692290389, 'rms': 0.0016245734697727086, 'p95': 0.003589221437393534, 'p99': 0.004314214838795581, 'hausdorff': 0.004984187107782277, 'trimmed_hausdorff_99': 0.004314214838795581, 'count': 150000}, 'gt_to_recon': {'mean': 0.0014192314397015037, 'median': 0.0005025347886563891, 'rms': 0.00365355406218693, 'p95': 0.008369692823293006, 'p99': 0.01900580682101962, 'hausdorff': 0.02506879589093513, 'trimmed_hausdorff_99': 0.01900580682101962, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.18455333333333335, 'recall': 0.26598666666666665, 'fscore': 0.21791062261089164}, 0.0005: {'tau_m': 0.0005, 'precision': 0.35247333333333336, 'recall': 0.49803333333333333, 'fscore': 0.4127973971070839}, 0.001: {'tau_m': 0.001, 'precision': 0.5826, 'recall': 0.77636, 'fscore': 0.6656668864425738}, 0.002: {'tau_m': 0.002, 'precision': 0.7972533333333334, 'recall': 0.91212, 'fscore': 0.8508272549004313}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 30.548807668802716, 'median': 25.172730329470397, 'p90': 64.84502769165297, 'p95': 77.54242650234958, 'max': 89.99958149155106, 'count': 146833}, 'gt_to_recon': {'mean': 25.811757206282266, 'median': 20.818208131884454, 'p90': 52.80056920111992, 'p95': 71.60594087303937, 'max': 89.99952915416641, 'count': 139175}, 'symmetric': {'mean': 28.243700825213434, 'median': 22.83051012586612, 'p90': 59.801394255892504, 'p95': 75.47912215435814, 'max': 89.99958149155106, 'count': 286008}}, 'recon_topology': {'boundary_edges': 1229, 'connected_components_number': 214, 'edges_number': 291193, 'faces_number': 193719, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 3, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 1, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 96442}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}

tsdf hi sm
{'recon_to_gt': {'mean': 0.0011122167091761846, 'median': 0.0007257485592908369, 'rms': 0.0015496631621419014, 'p95': 0.0035535577037880037, 'p99': 0.0041375111596545595, 'hausdorff': 0.004703952119260751, 'trimmed_hausdorff_99': 0.0041375111596545595, 'count': 150000}, 'gt_to_recon': {'mean': 0.0015001299626536868, 'median': 0.0005756616151867998, 'rms': 0.0037184266838952134, 'p95': 0.00853440775193787, 'p99': 0.019280002614545523, 'hausdorff': 0.025371043246496614, 'trimmed_hausdorff_99': 0.019280002614545523, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.18948666666666666, 'recall': 0.22415333333333334, 'fscore': 0.20536731435816438}, 0.0005: {'tau_m': 0.0005, 'precision': 0.37512, 'recall': 0.44499333333333335, 'fscore': 0.4070800764122032}, 0.001: {'tau_m': 0.001, 'precision': 0.61752, 'recall': 0.73352, 'fscore': 0.6705401326385599}, 0.002: {'tau_m': 0.002, 'precision': 0.8222866666666667, 'recall': 0.90714, 'fscore': 0.8626316931237328}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 17.610539239405185, 'median': 11.481946302482953, 'p90': 41.788697366785705, 'p95': 59.23027555112158, 'max': 89.99501673400094, 'count': 147670}, 'gt_to_recon': {'mean': 16.01854861094305, 'median': 10.31549246166262, 'p90': 35.814485282487254, 'p95': 54.33656094192172, 'max': 90.0, 'count': 138785}, 'symmetric': {'mean': 16.839233382041492, 'median': 10.895786201881894, 'p90': 39.06494465815975, 'p95': 57.33083988683372, 'max': 90.0, 'count': 286455}}, 'recon_topology': {'boundary_edges': 1229, 'connected_components_number': 214, 'edges_number': 291193, 'faces_number': 193719, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 3, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 1, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 96442}, 'gt_topology': {'boundary_edges': 24760, 'connected_components_number': 34, 'edges_number': 798812, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 6, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 274552}}



unc nrm
{'mse': 0.005598970613081594, 'iou': 0.9596285094706302}

van nrm
{'mse': 0.005642056479972954, 'iou': 0.9594091161544601}

unc nonorm
{'mse': 0.005896224758166675, 'iou': 0.937123829754161}

tsdf lo
{'mse': 0.004088319433641242, 'iou': 0.9711199871352294}

tsdf lo smth
{'mse': 0.00644710582133027, 'iou': 0.9572946071768417}

tsdf hi
{'mse': 0.002875181111001981, 'iou': 0.9727542761664726}

tsdf hi smth
{'mse': 0.003802924039568613, 'iou': 0.973295019912828}