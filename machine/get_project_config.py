def get_project_config(project_info):
    name = project_info['name']

    dict_config_name = project_info['dictionary_config']
    dict_config = project_info[name][dict_config_name]

    topic_method = project_info['topic_method']
    topic_config_name = project_info['topic_config']
    topic_config = project_info[name][topic_config_name]

    train_method = project_info['train_method']
    train_config_name = project_info['train_config']
    train_config = project_info[name][train_config_name]

    return name, dict_config, topic_method, topic_config, train_method, train_config