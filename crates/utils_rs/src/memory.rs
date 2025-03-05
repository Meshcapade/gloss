pub fn get_last_relevant_func_name(cb_stack: &str) -> String {
    let stack_parts = cb_stack.split('\n');
    for f in stack_parts {
        //split a string like number:package:name:func into number and the rest (package:name:func)
        let mut parts = f.splitn(2, ':');
        let opt_func_number = parts.next();
        let name_str = parts.next().unwrap_or("").trim().to_string();
        if let Some(func_number) = opt_func_number {
            if let Ok(_number) = func_number.trim().parse::<i32>() {
                //skip likely unrelevant lines that too low level
                if name_str.starts_with("at alloc")
                    || name_str.starts_with("alloc")
                    || name_str.starts_with("at <alloc")
                    || name_str.starts_with("at re_memory")
                    || name_str.starts_with("at<re_memory")
                    || name_str.starts_with("<re_memory")
                    || name_str.starts_with("at std::thread")
                    || name_str.starts_with("at <u8 as alloc::vec")
                    || name_str.starts_with("at <T as alloc::vec")
                    || name_str.starts_with("at __rust_alloc")
                    || name_str.starts_with("at __rust_realloc")
                    || name_str.starts_with("Error")
                    || name_str.starts_with("at imports.wbg")
                {
                    continue;
                }

                return name_str;
            }
        }
    }
    "unkown".to_string()
}
