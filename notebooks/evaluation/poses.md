
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

TSDF 1/1000, 4: 0.1s
SPF_unc: 26.3s
SPF_vanla: 24.7

SPF_unc:
{'recon_to_gt': {'mean': 0.0002183763525769775, 'median': 0.00016369076143157857, 'rms': 0.0002972716232881154, 'p95': 0.0005990164350735456, 'p99': 0.0009164342672432511, 'hausdorff': 0.0029240677672420223, 'trimmed_hausdorff_99': 0.0009164342672432511, 'count': 150000}, 'gt_to_recon': {'mean': 0.0002577214576348356, 'median': 0.00017512056514538965, 'rms': 0.00041084034355963115, 'p95': 0.0007226775211187899, 'p99': 0.0014192842128586398, 'hausdorff': 0.005996769479851033, 'trimmed_hausdorff_99': 0.0014192842128586398, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.6695266666666667, 'recall': 0.6374933333333334, 'fscore': 0.6531174526616103}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9145266666666667, 'recall': 0.8819933333333333, 'fscore': 0.8979654255511271}, 0.001: {'tau_m': 0.001, 'precision': 0.9935866666666666, 'recall': 0.9781866666666666, 'fscore': 0.9858265279083239}, 0.002: {'tau_m': 0.002, 'precision': 0.9996866666666666, 'recall': 0.9949733333333334, 'fscore': 0.9973244312536694}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 9.401015366476482, 'median': 5.845446002993283, 'p90': 21.4653349776307, 'p95': 30.449905384287092, 'max': 89.9947702877462, 'count': 150000}, 'gt_to_recon': {'mean': 9.852630528610113, 'median': 6.039855433377266, 'p90': 22.553486158542352, 'p95': 31.761664022594868, 'max': 89.93229080004195, 'count': 149855}, 'symmetric': {'mean': 9.626713754435778, 'median': 5.944225019059403, 'p90': 22.01844442595976, 'p95': 31.151956273291923, 'max': 89.9947702877462, 'count': 299855}}, 'recon_topology': {'boundary_edges': 434, 'connected_components_number': 1, 'edges_number': 161644, 'faces_number': 107618, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 238, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 44, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 53925}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}

SPF_van
{'recon_to_gt': {'mean': 0.0002562816068535004, 'median': 0.00015439784762386055, 'rms': 0.0004986943494632691, 'p95': 0.0007390384173496772, 'p99': 0.0015957819761470025, 'hausdorff': 0.009786568154053333, 'trimmed_hausdorff_99': 0.0015957819761470025, 'count': 150000}, 'gt_to_recon': {'mean': 0.000300460266994677, 'median': 0.00016922310718787276, 'rms': 0.0005180606383371225, 'p95': 0.0010013361308841132, 'p99': 0.002229802839733982, 'hausdorff': 0.005447263636536984, 'trimmed_hausdorff_99': 0.002229802839733982, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.6715333333333333, 'recall': 0.6345533333333333, 'fscore': 0.6525198150879897}, 0.0005: {'tau_m': 0.0005, 'precision': 0.88462, 'recall': 0.84594, 'fscore': 0.8648477288276626}, 0.001: {'tau_m': 0.001, 'precision': 0.9744733333333333, 'recall': 0.9499266666666667, 'fscore': 0.9620434476084898}, 0.002: {'tau_m': 0.002, 'precision': 0.9935866666666666, 'recall': 0.98598, 'fscore': 0.9897687186758045}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 10.482681753039737, 'median': 6.046004582748313, 'p90': 24.997068612593548, 'p95': 35.931565261054544, 'max': 89.98759280710578, 'count': 149568}, 'gt_to_recon': {'mean': 10.996869906077478, 'median': 6.306668740322015, 'p90': 26.266622408114827, 'p95': 38.001862662387126, 'max': 89.99757991787723, 'count': 149864}, 'symmetric': {'mean': 10.740029976899738, 'median': 6.1780852833766495, 'p90': 25.623416513225994, 'p95': 36.958940218833085, 'max': 89.99757991787723, 'count': 299432}}, 'recon_topology': {'boundary_edges': 621, 'connected_components_number': 8, 'edges_number': 159138, 'faces_number': 105885, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 567, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 102, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 53044}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}

TSDF
{'recon_to_gt': {'mean': 0.00014687244991346725, 'median': 0.00010973439865250012, 'rms': 0.00020372690206165022, 'p95': 0.00041298952836230074, 'p99': 0.0006508213395319548, 'hausdorff': 0.002057437845167942, 'trimmed_hausdorff_99': 0.0006508213395319548, 'count': 150000}, 'gt_to_recon': {'mean': 0.00014291060112843818, 'median': 0.00010536931534717298, 'rms': 0.00021130080635847935, 'p95': 0.0003896392379367563, 'p99': 0.0006481079781228031, 'hausdorff': 0.0032925627570274496, 'trimmed_hausdorff_99': 0.0006481079781228031, 'count': 150000}, 'fscore': {0.00025: {'tau_m': 0.00025, 'precision': 0.8345666666666667, 'recall': 0.85004, 'fscore': 0.8422322710345836}, 0.0005: {'tau_m': 0.0005, 'precision': 0.9732933333333333, 'recall': 0.97732, 'fscore': 0.9753025105266035}, 0.001: {'tau_m': 0.001, 'precision': 0.99784, 'recall': 0.99614, 'fscore': 0.9969892753187092}, 0.002: {'tau_m': 0.002, 'precision': 0.9999933333333333, 'recall': 0.99932, 'fscore': 0.999656553283294}}, 'normal_error_deg_triangle': {'recon_to_gt': {'mean': 16.716263678836704, 'median': 13.78740402850657, 'p90': 32.13332990752431, 'p95': 40.4301692097526, 'max': 89.94411912570389, 'count': 150000}, 'gt_to_recon': {'mean': 16.48610273028965, 'median': 13.68076257380698, 'p90': 31.403478987098836, 'p95': 39.502785143164324, 'max': 90.0, 'count': 150000}, 'symmetric': {'mean': 16.601183204563178, 'median': 13.731451612698914, 'p90': 31.780593280606666, 'p95': 39.95722069599961, 'max': 90.0, 'count': 300000}}, 'recon_topology': {'boundary_edges': 709, 'connected_components_number': 29, 'edges_number': 277883, 'faces_number': 185019, 'genus': 1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 0, 'is_mesh_two_manifold': True, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 0, 'number_holes': 97, 'unreferenced_vertices': 0, 'vertices_number': 92823}, 'gt_topology': {'boundary_edges': 8260, 'connected_components_number': 43, 'edges_number': 790562, 'faces_number': 524288, 'genus': -1, 'incident_faces_on_non_two_manifold_edges': 0, 'incident_faces_on_non_two_manifold_vertices': 10, 'is_mesh_two_manifold': False, 'non_two_manifold_edges': 0, 'non_two_manifold_vertices': 4, 'number_holes': -1, 'unreferenced_vertices': 0, 'vertices_number': 266305}}

SPF_unc
{'mse': 0.0014879863013830294, 'iou': 0.9819315346695654}

SPF_van
{'mse': 0.001781667522806911, 'iou': 0.978447900854623}

TSDF
{'mse': 0.0019712958440211096, 'iou': 0.9892930066541767}


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