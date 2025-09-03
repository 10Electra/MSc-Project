
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