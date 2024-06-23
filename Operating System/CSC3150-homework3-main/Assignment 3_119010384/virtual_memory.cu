#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ u32 vm_vpage_num(VirtualMemory *vm, u32 addr) 
{
	return (u32)(addr / vm->PAGESIZE);
}

__device__ u32 vm_addr_page_offest(VirtualMemory *vm, u32 addr) 
{
	return (u32)(addr % vm->PAGESIZE);
}

__device__ u32 vm_get_physical_addr(VirtualMemory *vm, u32 pp_num, u32 page_offest) 
{
	return (u32)(pp_num * vm->PAGESIZE + page_offest);
}

__device__ u32 vm_find_vaddr_pp_num(VirtualMemory *vm, u32 addr) 
{
	u32 i;
  	u32 vp_num = vm_vpage_num(vm, addr);                    
  	
  	for (i = 0; i < vm->PAGE_ENTRIES; i++) {
    	if (vm->invert_page_table[i] == vp_num) {
     	 	return i;
    	}
  	}
	                       
  	return (u32)-1;
}

__device__ u32 vm_find_physical_page_num(VirtualMemory *vm)
{
	u32 i;
	u32 max = vm->PAGE_ENTRIES << 1;
	u32 min_index = 0xFFFFFFFF;
	u32 page_num = 0;
	
  	for (i = vm->PAGE_ENTRIES; i < max; i++) {
      	if (vm->invert_page_table[i] < min_index) {
          	min_index = vm->invert_page_table[i];
          	page_num = i;
      	}
  	}

	return page_num;
}

__device__ void vm_copy_page_data(VirtualMemory *vm, u32 pp_num, u32 vp_num)
{
	int i;
	u32 poffest = pp_num * vm->PAGESIZE;
	u32 voffest = vp_num * vm->PAGESIZE;
	
	for (i = 0; i < vm->PAGESIZE; i++){
    	vm->buffer[poffest + i] = vm->storage[voffest + i];
  	}

	return;
}

__device__ u32 vm_do_page_fault(VirtualMemory *vm, int vp_num)
{
  	int pp_num;

	pp_num = vm_find_physical_page_num(vm);
  	pp_num -= vm->PAGE_ENTRIES;
	
	vm_copy_page_data(vm, pp_num, vp_num);
  
  	vm->invert_page_table[pp_num] = vp_num;

	*vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
	
  	return pp_num;
} 

__device__ void vm_do_lru(VirtualMemory *vm, u32 pp_num)
{ 
	u32 i;
  	u32 pp_index = pp_num + vm->PAGE_ENTRIES;
	u32 max = vm->PAGE_ENTRIES << 1;
	
  	for (i = vm->PAGE_ENTRIES; i < max; i++) {
    	if (vm->invert_page_table[i] > vm->invert_page_table[pp_index]){
      		vm->invert_page_table[i] = vm->invert_page_table[i] - 1;
    	}
  	}
	
	vm->invert_page_table[pp_index] = vm->PAGE_ENTRIES - 1;

	return;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) 
{
	u32 vp_num = vm_vpage_num(vm, addr);
	u32 page_offest = vm_addr_page_offest(vm, addr);
	u32 pp_num = vm_find_vaddr_pp_num(vm, addr);
	u32 physical_addr;
	uchar read_data;
	
	//not find
	if (pp_num == (u32)-1) {
		//find a physical page, do page table mapping
		pp_num = vm_do_page_fault(vm, vp_num);
		
    	physical_addr = vm_get_physical_addr(vm, pp_num, page_offest);
		//do lru update
    	vm_do_lru(vm, pp_num);
		
    	return vm->buffer[physical_addr];
	}

    physical_addr = vm_get_physical_addr(vm, pp_num, page_offest);
    vm_do_lru(vm, pp_num);
    return vm->buffer[physical_addr];
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) 
{
	u32 vp_num = vm_vpage_num(vm, addr);
	u32 page_offest = vm_addr_page_offest(vm, addr);
	u32 pp_num = vm_find_vaddr_pp_num(vm, addr);
	u32 physical_addr;

	//not find
	if (pp_num == (u32)-1) {
		//find a physical page, do page table mapping
		pp_num = vm_do_page_fault(vm, vp_num);
    	physical_addr = vm_get_physical_addr(vm, pp_num, page_offest);
		//do lru update
    	vm_do_lru(vm, pp_num);
		
		vm->buffer[physical_addr] = value;
    	vm->storage[addr] = value;

		return;
	}

    physical_addr = vm_get_physical_addr(vm, pp_num, page_offest);
    vm_do_lru(vm, pp_num);
    vm->buffer[physical_addr] = value;
    vm->storage[addr] = value;

	return;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) 
{
	int i;
	uchar read_data;
	
	printf("snapshot:\n");
	for (i = 0; i < input_size; i++) {
		read_data = vm_read(vm, i+offset);
		results[i] = read_data;
	}
	
	return;
}

