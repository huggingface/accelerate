import torch


def prepare_nd_device_mesh(tp_size=1, use_fsdp=False):
    """Returns a multi dimensional device mesh.
    Extend this function to support various combinations of parallelisms.
    """
    from torch.distributed.device_mesh import init_device_mesh

    mesh_dim_names = ()
    mesh_dims = ()
    if tp_size <= 1 and not use_fsdp:
        return None
    if tp_size > 1:
        mesh_dim_names = ("tp",)
        mesh_dims = (tp_size,)
    if use_fsdp:
        num_nodes = torch.distributed.get_world_size() // torch.cuda.device_count()
        nproc_per_node = torch.cuda.device_count()
        mesh_dim_names = (
            "dp",
            "fsdp",
        ) + mesh_dim_names
        mesh_dims = (
            num_nodes,
            nproc_per_node // tp_size,
        ) + mesh_dims
    device = "cuda"
    return init_device_mesh(device, mesh_dims, mesh_dim_names=mesh_dim_names)
