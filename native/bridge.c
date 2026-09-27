/* Included after the upstream environment header/source. No trainer or Python. */
#include <stdint.h>
#ifdef _WIN32
#define FG_API __declspec(dllexport)
#else
#define FG_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
#define FG_EXTERN extern "C" FG_API
#else
#define FG_EXTERN FG_API
#endif
#if PUF_BACKEND == PUF_GPU
#define FG_GPU 1
#else
#define FG_GPU 0
#endif

typedef struct {
    Env* envs;
    int count;
    Ini ini; /* Some environments retain pointers into their configuration. */
} FgBatch;

static const int fg_actions[] = ACT_SIZES;
static char fg_message[256];
static int fg_busy; /* GPU environments, CPU Admiral and Chess keep global state. */
static int fg_fail(const char* message) {
    snprintf(fg_message, sizeof(fg_message), "%s", message);
    return -1;
}
FG_EXTERN const char* fg_error(void) { return fg_message; }
FG_EXTERN int fg_info(int key, int index) {
    switch (key) {
        case 0: return 1;
        case 1: return FG_GPU;
        case 2: return sizeof(obs_t)==1 ? 1 : sizeof(obs_t)==2 ? 2 : sizeof(obs_t)==4 ? 3 : 4;
        case 3: return OBS_SIZE;
        case 4: return NUM_ATNS;
        case 5: return index>=0 && index<NUM_ATNS ? fg_actions[index] : 0;
        case 6: { int n=0; for(int i=0;i<NUM_ATNS;i++) n+=fg_actions[i]; return n; }
        case 7:
#ifdef PUFFER_ADMIRAL
            return N_TEAMS;
#else
            return 1;
#endif
        case 8:
#if FG_GPU || defined(MY_VEC_INIT)
            return 0;
#else
            return 1;
#endif
        default: return -1;
    }
}

static void fg_ini_free(Ini* ini) {
    for(int i=0;i<ini->num_sections;i++) dict_clear(&ini->sections[i]);
    free(ini->sections);
}

#if !FG_GPU
static void fg_envs_free(FgBatch* b) {
    for(int i=0;i<b->count;i++) puf_close(&b->envs[i]);
#if defined(MY_VEC_CLOSE) && defined(PUFFER_CHESS)
    my_vec_close(b->envs); /* Global FEN cache can exist even with zero complete games. */
#elif defined(MY_VEC_CLOSE)
    if(b->count>0) my_vec_close(b->envs);
#endif
    free(b->envs);
}
#endif

FG_EXTERN void* fg_create(int n, unsigned int seed, void* obs, float* actions,
        float* rewards, float* terminals, unsigned char* masks, const char* config) {
    if(n<=0 || n%fg_info(7,0)) { fg_fail("invalid agent count"); return NULL; }
    if(fg_busy) { fg_fail("upstream backend permits only one live batch per library"); return NULL; }
    FgBatch* b = (FgBatch*)calloc(1,sizeof(FgBatch));
    if(!b) { fg_fail("batch allocation failed"); return NULL; }
    puf_ini_load_file(&b->ini, config);
    Dict* ek = puf_ini_section(&b->ini, "env", 1);
#if FG_GPU
    b->envs = puf_vec_create(n, ek, (obs_t*)obs, actions, rewards, terminals);
    cudaError_t status = cudaDeviceSynchronize();
    if(!b->envs || status!=cudaSuccess) {
        if(b->envs) puf_close(b->envs);
        fg_fail(cudaGetErrorString(status)); fg_ini_free(&b->ini); free(b); return NULL;
    }
#else
#ifdef MY_VEC_INIT
    Dict* vk = puf_ini_section(&b->ini, "vec", 1);
    dict_set(vk,"total_agents",n); dict_set(vk,"num_buffers",1);
    dict_set(vk,"num_threads",1); dict_set(vk,"num_policies",1);
    dict_set(vk,"hist_policy_percent",0);
    int start=0, count=0;
    b->envs = my_vec_init(&b->count, &start, &count, vk, ek);
#else
    b->envs = (Env*)calloc((size_t)n,sizeof(Env));
    if(!b->envs) { fg_fail("environment allocation failed"); fg_ini_free(&b->ini); free(b); return NULL; }
    int slots=0;
    while(slots<n) {
        Env* e=&b->envs[b->count++];
        e->rng=seed+b->count-1;
        puf_init(e,ek);
        if(e->num_agents<=0 || e->num_agents>n-slots) break;
        slots+=e->num_agents;
    }
#endif
    int total=0;
    for(int i=0;i<b->count;i++) total+=b->envs[i].num_agents;
    if(total!=n) {
        fg_envs_free(b); fg_ini_free(&b->ini); free(b);
        fg_fail("agent count must fit complete environments"); return NULL;
    }
    int k=0, mask_size=fg_info(6,0);
    for(int i=0;i<b->count;i++) {
        Env* e=&b->envs[i];
        e->tag=0; e->boundary_reached=0;
        for(int j=0;j<e->num_agents;j++,k++) {
            Agent* a=&e->agents[j];
            a->observations=(obs_t*)obs+(size_t)k*OBS_SIZE;
            a->actions=actions+(size_t)k*NUM_ATNS;
            a->rewards=rewards+k; a->terminals=terminals+k;
            a->action_mask=masks+(size_t)k*mask_size; a->policy=0;
        }
    }
#endif
#if FG_GPU || defined(PUFFER_ADMIRAL) || defined(PUFFER_CHESS)
    fg_busy = 1;
#endif
    return b;
}

FG_EXTERN int fg_reset(void* handle, void* stream) {
    if(!handle) return fg_fail("closed batch");
    FgBatch* b=(FgBatch*)handle;
#if FG_GPU
    /* Upstream Breakout/Robot Arm reset launches on stream 0, ignoring binding.
       Reset is intentionally a synchronization boundary; step remains async. */
    cudaError_t s=cudaStreamSynchronize((cudaStream_t)stream);
    if(s!=cudaSuccess) return fg_fail(cudaGetErrorString(s));
    puf_bind_stream((cudaStream_t)stream);
    puf_reset(b->envs);
    s=cudaDeviceSynchronize();
    return s==cudaSuccess ? 0 : fg_fail(cudaGetErrorString(s));
#else
    for(int i=0;i<b->count;i++) puf_reset(&b->envs[i]);
    return 0;
#endif
}

static void fg_advance(FgBatch* b) {
#if FG_GPU
    puf_step(b->envs);
#else
    #pragma omp parallel for schedule(static) if(b->count>=256)
    for(int i=0;i<b->count;i++) puf_step(&b->envs[i]);
#endif
}
FG_EXTERN int fg_step(void* handle, void* stream) {
    if(!handle) return fg_fail("closed batch");
#if FG_GPU
    puf_bind_stream((cudaStream_t)stream);
#endif
    fg_advance((FgBatch*)handle);
#if FG_GPU
    cudaError_t s=cudaGetLastError();
    return s==cudaSuccess ? 0 : fg_fail(cudaGetErrorString(s));
#else
    return 0;
#endif
}

/* Reference for benchmarks: loop entirely in native code, identical kernels. */
FG_EXTERN int fg_steps(void* handle, void* stream, int steps) {
    if(!handle || steps<0) return fg_fail("invalid native loop");
#if FG_GPU
    puf_bind_stream((cudaStream_t)stream);
#endif
    for(int i=0;i<steps;i++) fg_advance((FgBatch*)handle);
#if FG_GPU
    cudaError_t s=cudaGetLastError();
    return s==cudaSuccess ? 0 : fg_fail(cudaGetErrorString(s));
#else
    return 0;
#endif
}
FG_EXTERN int fg_close(void* handle) {
    if(!handle) return 0;
    FgBatch* b=(FgBatch*)handle;
#if FG_GPU
    puf_close(b->envs);
#else
    fg_envs_free(b);
#endif
    fg_busy=0;
    fg_ini_free(&b->ini); free(b);
    return 0;
}
