/* Exercise the upstream vector lifecycle contract, including failed creation. */
#include <assert.h>
typedef float obs_t;
#include "pufferenv.h"
#define OBS_SIZE 1
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define MY_VEC_INIT
#define MY_VEC_CLOSE
struct Env { Agent agents[2]; int num_agents, tag, boundary_reached; int* shared; };
static int shared_batches;

Env* my_vec_init(int* size, int* start, int* count, Dict* vk, Dict* ek) {
    Env* envs = calloc(2,sizeof(Env));
    int* shared = calloc(1,sizeof(int));
    shared_batches++;
    for(int i=0;i<2;i++) { envs[i].num_agents=2; envs[i].shared=shared; }
    *size=2;
    return envs;
}
void puf_reset(Env* e) {}
void puf_step(Env* e) {}
void puf_close(Env* e) { ++*e->shared; }
void my_vec_close(Env* envs) {
    assert(*envs[0].shared==2); /* Per-environment close must run first. */
    free(envs[0].shared);
    shared_batches--;
}
#include "../native/bridge.c"

int main(int argc, char** argv) {
    float obs[4], actions[4], rewards[4], terminals[4];
    unsigned char masks[8];
    void* a=fg_create(4,0,obs,actions,rewards,terminals,masks,argv[1]);
    assert(a && shared_batches==1);
    fg_close(a);
    assert(shared_batches==0);
    /* Three slots cannot hold two-agent games. The failed batch still owns resources. */
    assert(!fg_create(3,0,obs,actions,rewards,terminals,masks,argv[1]));
    assert(shared_batches==0);
    /* Per-batch shared resources do not imply a process-wide singleton. */
    a=fg_create(4,0,obs,actions,rewards,terminals,masks,argv[1]);
    void* b=fg_create(4,0,obs,actions,rewards,terminals,masks,argv[1]);
    assert(a && b && shared_batches==2);
    fg_close(a); fg_close(b);
    assert(shared_batches==0);
}
